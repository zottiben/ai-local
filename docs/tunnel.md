# Reaching the gateway from outside the house

`llm.example.com` -> Cloudflare -> the tunnel on the tunnel host -> this PC over the LAN.

```
internet -> llm.example.com (Cloudflare edge)
         -> cloudflared on the tunnel host (10.0.0.10)
         -> http://10.0.0.20:8081  (ailocal gateway, bearer-authenticated)
         -> http://127.0.0.1:8080      (llama-server, loopback only)
```

The tunnel terminates on the tunnel host rather than here: it already runs, it needs no new
daemon on this machine, and the tunnel host only shuffles bytes so its weak CPU is irrelevant.

## What is exposed

Only the gateway binds `0.0.0.0`. llama-server stays on loopback, so the authenticated
gateway is the sole way in. Every route except `/health` requires the bearer key from
`~/.config/ailocal/gateway.key`.

## Adding the route

the tunnel host's tunnel is **remotely managed** - it runs with `TUNNEL_TOKEN` and no local
config file, so ingress comes from the Zero Trust dashboard and cannot be edited over
SSH. In *Networks -> Tunnels -> (the tunnel) -> Published application routes*, add:

| Field | Value |
| --- | --- |
| Subdomain | `lara` |
| Domain | `benzotti.me` |
| Service | `HTTP` |
| URL | `10.0.0.20:8081` |

Every existing route points at `10.0.0.10` because those services run *on* the tunnel host,
where `10.0.0.10` is effectively localhost. This one cannot: gemma4 needs 11.3 GB of
VRAM and the tunnel host has no GPU, so cloudflared makes one LAN hop to the machine that does.
It forwards to any address it can route to, so this is no different to it.

### Pin the address first

This PC is on **DHCP** (`ipv4.method:auto`), and the tunnel host cannot resolve it by name -
there is no PTR record and no mDNS. So the route has to name an IP, and that IP has to
stop moving, or a lease change silently breaks the tunnel with no error anywhere except
a 502 at the edge.

Pin it on the router (10.0.0.1) as a DHCP reservation:

| | |
| --- | --- |
| MAC | `aa:bb:cc:dd:ee:ff` (enp4s0) |
| IP | `10.0.0.20` |

A reservation is preferable to a static address on the host: it survives an OS
reinstall, and it keeps all addressing in one place rather than half in the router and
half in NetworkManager.

## Why there is no Access policy on it

**A Cloudflare Access policy would break this.** Access authenticates with a browser
redirect, and a coding harness cannot complete one - it sends `Authorization: Bearer`
and expects an answer. Access *service tokens* exist for exactly this, but they need
`CF-Access-Client-Id` and `CF-Access-Client-Secret` headers, and neither Pi nor Claude
Code lets you set arbitrary headers on requests.

So the hostname is left without an Access application, and authentication is the
gateway's own bearer key: 32 bytes from `/dev/urandom`, compared in constant time.
That is the same posture as any API-key-protected endpoint.

If the key leaks, rotate it:

```
rm ~/.config/ailocal/gateway.key
ailocal gateway key                 # generates a new one
ailocal harness configure pi        # rewrite the harness configs
ailocal harness configure claude-code
systemctl --user restart ailocal-gateway.service
```

## Remote clients must stream

Cloudflare drops a proxied request that produces nothing for ~100 seconds (error 524).
A cold model load plus a long prompt can exceed that: Claude Code turns against this
model measured 1m53s-2m21s locally.

Streaming responses keep the connection fed, so they are unaffected - and Pi and Claude
Code both stream by default. A *non-streaming* request over the tunnel can time out
where the identical request over the LAN succeeds. Prefer `"stream": true` remotely.

## Verifying

```
ailocal gateway check                            # locally
ailocal gateway check https://llm.example.com   # through the tunnel
```

It asserts `/health` answers unauthenticated, that a missing or wrong credential is
refused, and that the real key is accepted. The second check is the one that matters:
if `/v1/models` returns 200 without a credential, the endpoint is open to the internet.
