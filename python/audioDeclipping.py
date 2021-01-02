from declipper import Declipper
import numpy as np
import scipy as sp
from stft import AnalysisFrame, SynthesisFrame
import matplotlib.pyplot as plt


prob, x = Declipper.clipper("quintet.wav",ptile=5)

A = AnalysisFrame(window="hamming",
                  overlap_percent=0.75)
D = SynthesisFrame(window="hamming",
                  overlap_percent=0.75)
xhatAna = prob.solve("ASPADE", A,iter_max=100)
xhatAnaBloc = prob.solve_block("ASPADE", A,4096,iter_max=100,progress=False)

plt.figure()
plt.plot(x)
plt.plot(xhatAna)
plt.plot(prob.y)
plt.legend(("x","\hat{x}","y"))
plt.show()

plt.figure()
plt.plot(x)
plt.plot(xhatAna)
plt.plot(prob.y)
plt.legend(("x","\hat{x}_{bloc}","y"))
plt.show()





# xhatSyn = prob.solve("SSPADE", D,iter_max=1000)


# clip,process = Declipper.sdr_study("quintet.wav",
#                                    'ASPADE', A,
#                                    ptiles=np.linspace(0,30,11),
#                                    iter_max=100)

# plt.plot(clip, clip-process)

# clip2,process2 = Declipper.sdr_study("quintet.wav",
#                                    'SSPADE', D,
#                                    ptiles=np.linspace(0,30,11),
#                                    iter_max=100)

# plt.plot(clip2, clip2-process2)

# sp.io.wavfile.write("hatAna",data=xhatAna, rate=44100)
# sp.io.wavfile.write("hatAna.wav",data=xhatAna, rate=44100)
# sp.io.wavfile.write("hatSyn.wav",data=xhatSyn, rate=44100)