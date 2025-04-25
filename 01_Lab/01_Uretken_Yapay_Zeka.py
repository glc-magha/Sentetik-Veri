"""Basit Bir Hikaye Üretici (Template + Rastgelelik ile)
import random

karakterler = ["Ali", "Zeynep", "Ayşe", "Mert"]
yerler = ["orman", "gizemli bir ada", "terk edilmiş bir şehir", "uzay gemisi"]
nesneler = ["sihirli bir taş", "zaman makinesi", "eski bir kitap", "kaybolmuş bir harita"]
olaylar = [
    "bir kapı arkasında büyük bir sır buldu",
    "geçmişe yolculuk yaptı",
    "gizemli bir varlıkla karşılaştı",
    "kendi geleceğini gördü"
]


def hikaye_uret():
    karakter = random.choice(karakterler)
    yer = random.choice(yerler)
    nesne = random.choice(nesneler)
    olay = random.choice(olaylar)

    hikaye = f"{karakter}, {yer} içinde gezerken {nesne} buldu ve {olay}."
    return hikaye


# 5 tane rastgele hikaye üretelim
for _ in range(5):
    print(hikaye_uret())"""