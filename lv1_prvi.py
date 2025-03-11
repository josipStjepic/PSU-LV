#Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je plaćen po radnom satu. Koristite ugrađenu
#Python metodu input(). Nakon toga izračunajte koliko je korisnik zaradio i ispišite na ekran. Na kraju prepravite
#rješenje na način da ukupni iznos izračunavate u zasebnoj funkciji naziva total_euro.
#Primjer:
#Radni sati: 35 h
#eura/h: 8.5
#Ukupno: 297.5 eura

def satnica(radni_sati,eura):
    return radni_sati * eura

radni_sati = float(input("Radni sati: "))
eura = float(input("eura/h: "))
ukupno  =  satnica(radni_sati,eura)
print(f"Ukupno: {ukupno} eura")