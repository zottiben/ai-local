# Reaching the gateway from outside your network

A Cloudflare tunnel is the least-effort way to expose the gateway without opening a
port or holding a static public IP.

```
internet -> <your-hostname>            (Cloudflare edge)
         -> cloudflared                (wherever it already runs)
         -> http://<gpu-host>:8081     (ailocal gateway, bearer-authenticated)
         -> http://127.0.0.1:8080      (llama-server, loopback only)
```

Run the tunnel wherever cloudflared already lives - typically a NAS or always-on server
rather than the GPU box. It only shuffles bytes, so it needs no GPU and barely any CPU,
and reusing an existing tunnel means one less daemon to maintain.

## What is exposed

Only the gateway binds a routable address:

```toml
# ~/.config/ailocal/config.toml
gateway_host = "0.0.0.0"
```

llama-server stays on loopback, so the authenticated gateway is the sole way in. Every
route except `/health` requires the bearer key from `~/.config/ailocal/gateway.key`.

## Adding the route

If cloudflared runs with a `TUNNEL_TOKEN` it is **remotely managed** - there is no local
`config.yml` to edit and ingress comes from the Zero Trust dashboard (the container logs
it as `Updated to new configuration ... version=N`). Add a public hostname there:

| Field | Value |
| --- | --- |
| Subdomain / domain | whatever you want to reach it at |
| Service | `HTTP` |
| URL | `<gpu-host-ip>:8081` |

Pointing at a *different* host from the one running cloudflared is fine - it forwards to
any address it can route to. That is the normal case here, since the GPU is rarely in the
same box as the tunnel.

### Pin the address first

The route names an IP, so that IP has to stop moving. If the GPU host is on DHCP, a lease
change breaks the tunnel with no error anywhere except a 502 at the edge. Check with:

```
nmcli -t connection show "<connection>" | grep ipv4.method   # "auto" means DHCP
```

Reserve it on the router against the host's MAC. A reservation is preferable to a static
address on the host: it survives an OS reinstall, and keeps all addressing in one place.

A hostname would be more robust than an IP, but only if the tunnel host can resolve it -
which needs a DNS entry or mDNS that many home routers do not provide. Check before
relying on it:

```
getent hosts <gpu-hostname>     # run this on the machine running cloudflared
```

## Why not to put Access in front of it

**A Cloudflare Access policy will break harness access.** Access authenticates with a
browser redirect, and a coding harness cannot complete one - it sends
`Authorization: Bearer` and expects an answer. Access *service tokens* exist for exactly
this, but they require `CF-Access-Client-Id` and `CF-Access-Client-Secret` headers, and
harnesses generally do not let you set arbitrary headers.

So leave the hostname without an Access application and let the gateway's own bearer key
be the authentication: 32 bytes from `/dev/urandom`, compared in constant time. That is
the same posture as any API-key-protected endpoint.

Rotate it if it leaks:

```
rm ~/.config/ailocal/gateway.key
ailocal gateway key                 # generates a new one
ailocal harness configure pi        # rewrite the harness configs
ailocal harness configure claude-code
systemctl --user restart ailocal-gateway.service
```

## Remote clients must stream

Cloudflare drops a proxied request that produces nothing for ~100 seconds (error 524).
A cold model load plus a long prompt can exceed that - Claude Code turns against a 12B
measured around two minutes.

Streaming responses keep the connection fed, so they are unaffected, and Pi and Claude
Code both stream by default. A *non-streaming* request over the tunnel can time out
where the identical request over the LAN succeeds. Prefer `"stream": true` remotely.

## Verifying

```
ailocal gateway check                          # locally
ailocal gateway check https://<your-hostname>  # through the tunnel
```

It asserts `/health` answers unauthenticated, that a missing or wrong credential is
refused, and that the real key is accepted. The 401 assertions are the ones that matter:
if `/v1/models` returns 200 without a credential, the endpoint is open to the internet.
