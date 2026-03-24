# Known Bugs

- **First client does not start training**: Occasionally the first client hangs and never begins its training round. **Workaround**: restart the federation.

- **CUDA MPS connection failed**: `CUDA_ERROR_MPS_CONNECTION_FAILED` from a dangling MPS server left by a previous crashed run. **Workaround**: restart the device.

- **Buggy visualization at map edge**: Clients or arcs near the antimeridian (+-180 longitude) may render incorrectly. **Status**: not fixed yet.

- **Client map positions randomized each round**: `_random_point()` picks a new random location within the country for each round's GIF, so clients visually jump around. **Status**: minor, no fix needed at this time.

- **All-or-nothing client config**: You must either define all client hardware profiles manually or let the sampler generate all of them. Partial manual configs are not supported yet.
