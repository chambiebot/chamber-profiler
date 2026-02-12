# Project Context
- Owner: chambiebot (GitHub)
- Company: Chamber (usechamber.io) - YC W26, GPU infrastructure optimization
- This is a GPU profiling tool like nsys but easier to use
- Must integrate with Chamber's platform
- Chamber concepts: Teams, Capacity Pools, Reservations, Reserved/Elastic Workloads
- Agent: Helm chart oci://public.ecr.aws/chamber/chamber-agent-chart
- WebSocket: wss://controlplane-api.usechamber.io/agent
- Dashboard: app.usechamber.io
- Related repos: chambiebot/job-duration-prediction, chambiebot/chamber-mle-agent
