#Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja nekakvu ocjenu i nalazi se između 0.0 i
#1.0. Ispišite kojoj kategoriji pripada ocjena na temelju sljedećih uvjeta:
#>= 0.9 A
#>= 0.8 B
#>= 0.7 C
#>= 0.6 D
#< 0.6 F
#Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe). Također, ako je
#broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovarajuću poruku
try:
    ocjena = float(input("Unesi ocjenu:"))

    if 0<= ocjena <= 1:
        if ocjena >= 0.9:
            print("A")
        elif ocjena >= 0.8:
            print("B")
        elif ocjena >= 0.7:
            print("C")
        elif ocjena >= 0.6:
            print("D")   
        else:   
            print("F")
    else:
        print("unesi u intervalu od 0 do 1")
except ValueError:
    print("Greška,ponovo unesi broj")
