#Za mtcars skup podataka napišite programski kod koji će odgovoriti na sljedeća pitanja:
#1. Kojih 5 automobila ima najveću potrošnju? (koristite funkciju sort)
#2. Koja tri automobila s 8 cilindara imaju najmanju potrošnju?
#3. Kolika je srednja potrošnja automobila sa 6 cilindara?
#4. Kolika je srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs?
#5. Koliko je automobila s ručnim, a koliko s automatskim mjenjačem u ovom skupu podataka?
#6. Koliko je automobila s automatskim mjenjačem i snagom preko 100 konjskih snaga?
#7. Kolika je masa svakog automobila u kilogramima?

import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')
mtcars['Weight_kg'] = mtcars['wt'] * 0.453592
#1.
print("1.")
najveca_potrosnja = mtcars.sort_values(by='mpg', ascending=False).head(5)
print(najveca_potrosnja[['car', 'mpg']])
#2.
print("2.")
najmanja_potrosnja_s8 = mtcars[mtcars['cyl'] == 8].sort_values(by='mpg').head(3)
print(najmanja_potrosnja_s8[['car', 'mpg']])
#3.
print("3.")
srednja_potrosnja_s6 = mtcars[mtcars['cyl'] == 6]['mpg'].mean()
print(srednja_potrosnja_s6)
#4.
print("4.")
prosnjecna_s4 = mtcars[(mtcars['cyl'] == 4) & (mtcars['wt'] >= 2000) & (mtcars['wt'] <= 2200)]['mpg'].mean()
print(prosnjecna_s4)
#5.
print("5.")
rucni = mtcars[mtcars['am'] == 1]['am'].count()
automatik = mtcars[mtcars['am'] == 0]['am'].count()
print("Rucni mjenjac: ", rucni)
print("Automatski mjenjac:", automatik)
#6.
print("6.")
automatik_100ks = mtcars[(mtcars['am'] == 0) & (mtcars['hp'] > 100)]['am'].count()
print(automatik_100ks)
#7.
print("7.")
masa = mtcars[['car', 'Weight_kg']]
print(masa)






