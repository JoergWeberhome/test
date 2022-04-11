import matplotlib.pyplot as plt
python_course_green = "#476042"
plt.figure(figsize=(6, 4))
plt.subplot(221) # äquivalent zu: plt.subplot(2, 2, 1)
plt.text(0.5, # x-Koordinate, 0 ganz links, 1 ganz rechts
         0.5, # y-Koordinate, 0 ganz oben, 1 ganz unten
         'subplot(2,2,1)', # der Text der ausgegeben wird
         horizontalalignment='center', # abgekürzt 'ha'
         verticalalignment='center', # abgekürzt 'va'
         fontsize=20, # kann auch  'font' genannt werden
         alpha=.5 # float (0.0 tranparent bis 1.0 undurchsichtig)
         )
plt.subplot(224, facecolor=python_course_green)
plt.text(0.5, 0.5,
         'subplot(2,2,4)',
         ha='center', va='center',
         fontsize=20,
         color="y")
plt.show()