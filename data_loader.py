import string
import spacy

class DataLoader:
    def __init__(self):
        # A small collection of Szymborska's poems (or fragments) for educational purposes.
        self.poems = [
            """Niektórzy lubią poezję
Niektórzy -
czyli nie wszyscy.
Nawet nie większość wszystkich ale mniejszość.
Nie licząc szkół, gdzie się musi,
i samych poetów,
będzie tych osób chyba dwie na tysiąc.

Lubią -
ale lubi się także rosół z makaronem,
lubi się komplementy i kolor niebieski,
lubi się stary szalik,
lubi się stawiać na swoim,
lubi się głaskać psa.

Poezję -
tylko co to takiego poezja.
Niejedna chwiejna odpowiedź
na to pytanie już padła.
A ja nie wiem i nie wiem i trzymam się tego
jak zbawiennej poręczy.""",
            
            """Nic dwa razy się nie zdarza
i nie zdarzy. Z tej przyczyny
zrodziliśmy się bez wprawy
i pomrzemy bez rutyny.

Choćbyśmy uczniami byli
najtępszymi w szkole świata,
nie będziemy repetować
żadnej zimy ani lata.

Żaden dzień się nie powtórzy,
nie ma dwóch podobnych nocy,
dwóch tych samych pocałunków,
dwóch jednakich spojrzeń w oczy.

Wczoraj, kiedy twoje imię
ktoś wymówił przy mnie głośno,
tak mi było, jakby róża
przez otwarte wpadła okno.

Dziś, kiedy jesteśmy razem,
odwróciłam twarz ku ścianie.
Róża? Jak wygląda róża?
Czy to kwiat? A może kamień?

Czemu ty się, zła godzino,
z niepotrzebnym mieszasz lękiem?
Jesteś - a więc musisz minąć.
Miniesz - a więc to jest piękne.

Uśmiechnięci, współobjęci
spróbujemy szukać zgody,
choć różnimy się od siebie
jak dwie krople czystej wody.""",

            """Kot w pustym mieszkaniu
Umrzeć - tego nie robi się kotu.
Bo co ma począć kot
w pustym mieszkaniu.
Wdrapywać się na ściany.
Ocierać się między meblami.
Nic niby tu nie zmienione,
a jednak pozamieniane.
Niby nie przesunięte,
a jednak porozsuwane.
I wieczorami lampa już nie świeci.

Słychać kroki na schodach,
ale to nie te.
Ręka, co kładzie rybę na talerzyk,
także nie ta, co kładła.

Coś się tu nie zaczyna
w swojej zwykłej porze.
Coś się tu nie odbywa
jak powinno.
Ktoś tutaj był i był,
a potem nagle zniknął
i uporczywie go nie ma.

Do wszystkich szaf się zajrzało.
Przez półki przebiegło.
Wcisnęło się pod dywan i sprawdziło.
Nawet złamało zakaz
i rozrzuciło papiery.
Co więcej jest do zrobienia.
Spać i czekać.

Niech no on tylko wróci,
niech no się pokaże.
Już on się dowie,
że tak z kotem nie można.
Będzie się szło w jego stronę
jakby się wcale nie chciało,
pomalutku,
na bardzo obrażonych łapach.
I żadnych skoków pisków na początek.""",

            """Cebula
Co innego cebula.
Ona nie ma wnętrzności.
Jest sobą na wskroś cebulą,
do stopnia cebuliczności.
Cebulasta na zewnątrz,
cebulowa do rdzenia,
mogłaby wejrzeć w siebie
cebula bez przerażenia.

W nas obczyzna i dzikość,
skóra ledwie tam taka,
piekło w nas internami,
anatomia gwałtowna,
a w cebuli cebula,
nie pokrętne jelita.
Ona wielokroć naga,
do głębi tumanu podobna.

Byt niesprzeczny cebula,
udany cebula twór.
W jednej po prostu druga,
w większej mniejsza zawarta,
a w następnej kolejna,
czyli trzecia i czwarta.
Dośrodkowa fuga.
Echo złożone w chór.

Cebula, to ja rozumiem:
najnadobniejszy brzuch świata.
Sam się aureolami
na własną chwałę oplata.
W nas - tłuszcze, nerwy, żyły,
śluzu i sekretności.
I jest nam odmówiony
idiotyzm doskonałości.""",

            """Radość pisania
Dokąd biegnie ta napisana sarna przez napisany las?
Czy z napisanej wody pić,
która jej pyszczek odbije jak kalka?
Dlaczego łeb podnosi, czy coś słyszy?
Na pożyczonych z prawdy czterech nóżkach wsparta
pod moimi palcami uchem strzyże.
Cisza - ten wyraz też szeleści po papierze
i rozgarnia
spowodowane słowem "las" gałęzie.

Nad białą kartką czają się do skoku
litery, które mogą ułożyć się źle,
zdania osaczające,
przed którymi nie będzie ratunku.

Jest w kropli atramentu spory zapas
myśliwych z przymrużonym okiem,
gotowych zbiec po stromym piórze w dół,
otoczyć sarnę, złożyć się do strzału.

Zapominają, że tu nie jest życie.
Inne, czarno na białym, panują tu prawa.
Oka mgnienie trwać będzie tak długo, jak zechcę,
pozwoli się podzielić na małe wieczności
pełne wstrzymanych w locie kul.
Na zawsze, jeśli każę, nic się tu nie stanie.
Bez mojej woli nawet liść nie spadnie
ani źdźbło się nie ugnie pod kropką kopytka.

Więc jest taki świat,
nad którym los sprawuję niezależny?
Czas, który wiążę łańcuchami znaków?
Istnienie na mój rozkaz nieustanne?

Radość pisania.
Możność utrwalania.
Zemsta ręki śmiertelnej."""
        ]
        
        try:
            self.nlp = spacy.load("pl_core_news_sm")
        except OSError:
            print("Downloading spacy model...")
            from spacy.cli import download
            download("pl_core_news_sm")
            self.nlp = spacy.load("pl_core_news_sm")

    def get_poems(self):
        return self.poems

    def clean_text(self, text):
        # Normalize text: lowercase, remove punctuation (except maybe sentence endings if we care about structure, 
        # but for simple n-grams usually we strip everything or keep minimal).
        # Let's strip punctuation for cleaner n-grams as per standard simple Markov models.
        
        text = text.lower()
        # Replace newlines with space to treat as continuous stream, or keep them?
        # Poems rely on lines. Let's replace newlines with a special token or just space.
        # For this exercise, let's treat it as a stream of words.
        text = text.replace('\n', ' ')
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize using spacy to handle Polish morphology better if needed, 
        # but for simple Markov, word splitting is often enough. 
        # However, let's use spacy for consistency with the plan.
        doc = self.nlp(text)
        tokens = [token.text for token in doc if not token.is_space]
        
        return tokens

if __name__ == "__main__":
    loader = DataLoader()
    poems = loader.get_poems()
    print(f"Loaded {len(poems)} poems.")
    sample_tokens = loader.clean_text(poems[0])
    print(f"Sample tokens from first poem: {sample_tokens[:10]}")
