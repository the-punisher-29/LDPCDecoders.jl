using BenchmarkTools
using LDPCDecoders

const SUITE = BenchmarkGroup()

# ── Setup ────────────────────────────────────────────────────────────────────

H = LDPCDecoders.parity_check_matrix(1000, 10, 9)
per = 0.01
err = rand(1000) .< per
syn = Bool.((H * err) .% 2)

# ── BP-OSD decode! ───────────────────────────────────────────────────────────

SUITE["bposd"] = BenchmarkGroup(["bposd"])
bposd0 = BeliefPropagationOSDDecoder(BitMatrix(Matrix(H)), per, 100; osd_order=0)
SUITE["bposd"]["decode_osd0"] = @benchmarkable decode!(d, s) setup=(d=deepcopy($bposd0); s=copy($syn)) evals=1
bposd2 = BeliefPropagationOSDDecoder(BitMatrix(Matrix(H)), per, 100; osd_order=2)
SUITE["bposd"]["decode_osd2"] = @benchmarkable decode!(d, s) setup=(d=deepcopy($bposd2); s=copy($syn)) evals=1

# ── BP decode! ───────────────────────────────────────────────────────────────

SUITE["bp"] = BenchmarkGroup(["bp"])
bp_decoder = BeliefPropagationDecoder(H, per, 100)
SUITE["bp"]["decode"] = @benchmarkable decode!(d, s) setup=(d=deepcopy($bp_decoder); s=copy($syn)) evals=1

# ── BitFlip decode! ──────────────────────────────────────────────────────────

SUITE["bitflip"] = BenchmarkGroup(["bitflip"])
bf_decoder = BitFlipDecoder(H, per, 100)
SUITE["bitflip"]["decode"] = @benchmarkable decode!(d, s) setup=(d=deepcopy($bf_decoder); s=copy($syn)) evals=1

# ── BP-OTS decode! ───────────────────────────────────────────────────────────

SUITE["bpots"] = BenchmarkGroup(["bpots"])
bpots_decoder = BPOTSDecoder(BitMatrix(Matrix(H)), per, 100; T=9, C=2.0)
SUITE["bpots"]["decode"] = @benchmarkable decode!(d, s) setup=(d=deepcopy($bpots_decoder); s=copy($syn)) evals=1

# ── SSF decode! ──────────────────────────────────────────────────────────────
# Use a smaller code for SSF since full-support enumeration on (1000,10,9) is expensive
H_ssf = LDPCDecoders.parity_check_matrix(100, 5, 4)
err_ssf = rand(100) .< per
syn_ssf = Bool.((H_ssf * err_ssf) .% 2)

SUITE["ssf"] = BenchmarkGroup(["ssf"])
ssf_decoder = SmallSetFlipDecoder(H_ssf, 50; t=2)
SUITE["ssf"]["decode"] = @benchmarkable decode!(d, s) setup=(d=deepcopy($ssf_decoder); s=copy($syn_ssf)) evals=1
