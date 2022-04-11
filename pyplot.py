import matplotlib.pyplot as plt
python_course_green = "#476042"
fig = plt.figure(figsize=(6, 4))
sub1 = fig.add_subplot(221) # alternativ: plt.subplot(2, 2, 1)
sub1.set_xticks([])
sub1.set_yticks([])

sub1.text(0.5, # x-Koordinate: 0 ganz links, 1 ganz rechts
          0.5, # y-Koordinate: 0 ganz oben, 1 ganz unten
          'subplot(2,2,1)', # der Text der ausgegeben wird
          horizontalalignment='center', # Abkürzung 'ha'
          verticalalignment='center', # sAbkürzung 'va'
          fontsize=20, #  'font' ist äquivalent
          alpha=.5 # Floatzahl von 0.0 transparent bis 1.0 opak
          )

sub2 = fig.add_subplot(223, facecolor=python_course_green)
sub2.set_xticks([])
sub2.set_yticks([])
sub2.text(0.5, 0.5,
          'subplot(2,2,3)',
          ha='center', va='center',
          fontsize=20,
          color="y")

plt.show()
