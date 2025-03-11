#Napišite program koji od korisnika zahtijeva unos brojeva u beskonačnoj petlji sve dok korisnik ne upiše „Done“ (bez
#navodnika). Pri tome brojeve spremajte u listu. Nakon toga potrebno je ispisati koliko brojeva je korisnik unio, njihovu
#srednju, minimalnu i maksimalnu vrijednost. Sortirajte listu i ispišite je na ekran.
#Dodatno: osigurajte program od pogrešnog unosa (npr. slovo umjesto brojke) na način da program zanemari taj unos i
#ispiše odgovarajuću poruku.


brojevi = []


while True:
    unos = input("Unesite broj: ")
    if unos.lower() == 'done':
        break
    try:
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print("Greška")
        
if len(brojevi) > 0:
    srednja = sum(brojevi) / len(brojevi)
    minimalna = min(brojevi)
    maksimalna = max(brojevi)
    print(f"\nUneseno {len(brojevi)}. brojeva")
    print(f"Srednja vrijednost: {srednja:.2f}")
    print(f"min: {minimalna}")
    print(f"Max: {maksimalna}")
    brojevi.sort()
    print("Sortirana lista brojeva:", brojevi)
else:
    print("Niste unijeli niti jedan broj.")