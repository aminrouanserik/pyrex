import matplotlib.pyplot as plt
from qcextender.dimensionlesswaveform import DimensionlessWaveform

e = DimensionlessWaveform.from_sim("SXS:BBH:1355")
qc = DimensionlessWaveform.from_sim("SXS:BBH:0180v2.0")

plt.plot(e.time, e[2, 2], label="SXS:BBH:1355, eccentric")
plt.plot(qc.time, qc[2, 2], label="SXS:BBH:0180v2.0, quasi-circular")
plt.xlim(max(e.time[0], qc.time[0]), min(e.time[-1], qc.time[-1]))
plt.ylabel("Mode [M/d]")
plt.xlabel("Time [M]")
plt.title("An eccentric and quasi circular SXS simulation")
plt.legend()
plt.show()
