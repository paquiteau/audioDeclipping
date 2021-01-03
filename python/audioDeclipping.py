from declipper import Declipper
import numpy as np

from stft import AnalysisFrame, SynthesisFrame
import matplotlib.pyplot as plt


prob, x = Declipper.clipper("../Sounds/double_bass.wav",ptile=5)

A = AnalysisFrame(window="hamming",
                  overlap_percent=0.75)
D = SynthesisFrame(window="hamming",
                  overlap_percent=0.75)

xhatAna = prob.solve("ASPADE", A,iter_max=500,k_init=100,progress=True)
xhatSyn = prob.solve("SSPADE", D,iter_max=500,k_init=100,progress=True)

# Figure 1
plt.figure()
plt.plot(prob.x)
plt.plot(xhatAna)
plt.plot(prob.y)
plt.legend(("x","xhat","y"))
plt.show()
plt.savefig("fig1.png")


#Figure 2
 
plt.figure()
plt.plot(clip, process-clip)
plt.plot(clip2, process2-clip)
plt.show()
plt.xlabel("SDR_clip")
plt.ylabel("SDR_process- SDR_clip")
plt.legend(("ASPADE","SSPADE"))
plt.savefig("../fig2.png")

#Figure 3
Abloc = AnalysisFrame(window="hamming",
                      length=512,
                      overlap_percent=0.75)
xhatAnaBloc = prob.solve_block("ASPADE", A, 4096,iter_max=500,progress=False)
