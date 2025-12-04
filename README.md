Model Markowa wyższego rzędu w analizie i
generowaniu tekstu
2025
1 Cel projektu
Projekt ma na celu zastosowanie łańcuchów Markowa wyższego rzędu (2. i
3. rzędu) do analizy i generowania tekstu na podstawie wybranego korpusu.
Projekt wykorzystuje metody przetwarzania języka naturalnego (NLP) oraz
elementy uczenia maszynowego do badania sekwencyjnych wzorców języko-
wych.
2 Zbiór danych
Do wyboru jeden z dwóch wariantów:
• Wariant A: Wiersze Wisławy Szymborskiej (minimum 5 utworów)
• Wariant B: Zbiór SMS-ów spam z Databricks (spam.csv)
3 Zadania
3.1 Krok 1: Przygotowanie i czyszczenie danych
• Usunięcie znaków specjalnych, tokenizacja
• Normalizacja tekstu (lowercase, usuwanie stop words)
• Podział na n-gramy dla potrzeb modeli wyższego rzędu
1
3.2 3.3 Krok 2: Analiza statystyczna tekstu
• Obliczenie dystrybucji n-gramów
• Analiza częstości występowania części mowy
• Porównanie charakterystyki językowej między wariantami
Krok 3: Implementacja łańcucha Markowa 2. i 3.
rzędu
• Zbudowanie macierzy przejść dla bigramów i trigramów
• Implementacja generatora tekstu wykorzystującego:
P(wt|wt−1,wt−2) = C(wt−2,wt−1,wt)
C(wt−2,wt−1) (1)
• Wygenerowanie po 10 przykładowych tekstów dla każdego rzędu łań-
cucha
3.4 Krok 4: Ewaluacja i porównanie modeli
• Ocena płynności generowanych tekstów
• Porównanie współczynnika powtórzeń
• Analiza zgodności ze stylem oryginalnego korpusu
3.5 Krok 5: Raportowanie
Przygotowanie kompleksowego raportu zawierającego:
• Opis implementacji modeli
• Porównanie wyników generacji dla różnych rzędów łańcucha
• Wizualizacje macierzy przejść
• Przykłady wygenerowanych tekstów
2
4 Wymagane rezultaty
• Kod implementujący łańcuchy Markowa 2. i 3. rzędu (Databricks No-
tebook)
• Raport PDF (6-10 stron) z analizą porównawczą
• Zbiór wygenerowanych tekstów dla obu modeli
• (Opcjonalnie) prezentacja podsumowująca wyniki
5 Metryki sukcesu
• Poprawna implementacja łańcuchów Markowa wyższego rzędu
• Jakość generowanych tekstów (spójność, zgodność ze stylem)
• Kompleksowość analizy porównawczej
• Przejrzystość raportu i dokumentacji
6 Ewaluacja wygenerowanych tekstów
6.1 Metryki jakości generowania tekstu
6.1.1 Płynność językowca (Fluency)
• Perpleksja (Perplexity) - mierzy zdolność modelu do przewidywania
kolejnych słów
PP(W) = exp−
N
N
i=1
log P(wi|wi−1,wi−2) (2)
gdzie W to sekwencja słów, N to długość sekwencji
• Self-BLEU - mierzy zróżnicowanie generowanych tekstów
1
Self-BLEU=
M
i=1
BLEU(si,{sj }j̸=i) (3)
1
M
3
6.1.2 Spójność semantyczna (Coherence)
• Współczynnik powtórzeń (Repetition Rate)
RR=
liczba powtórzonych n-gramów
liczba wszystkich n-gramów ×100% (4)
• Długość sekwencji bez powtórzeń
Unique Run Length = max{długość sekwencji bez powtórzonych słów}
(5)
6.1.3 Zgodność ze stylem korpusu (Style Consistency)
• Rozkład części mowy (POS Distribution Distance)
POS-Dist= JSD(Poriginal∥Pgenerated) (6)
gdzie JSD to dywergencja Jensena-Shannona
• Rozkład długości zdań
Sent-Len-Dist= KS-test(Loriginal,Lgenerated) (7)
test Kołmogorowa-Smirnowa dla rozkładów długości zdań
6.2 Metody obliczeniowe
6.2.1 Obliczanie perpleksji
def c a l c u l a t e _ p e r p l e x i t y ( model , t e s t _ t e x t ) :
t o k e n s = t o k e n i z e ( t e s t _ t e x t )
log_prob_sum = 0
f o r i in range ( 2 , len ( t o k e n s ) ) :
c o n t e x t = tuple ( t o k e n s [ i−2: i ] )
word = t o k e n s [ i ]
prob = model . g e t _ p r o b a b i l i t y ( c o n t e x t , word )
log_prob_sum += math . l o g ( prob ) i f prob > 0 e l s e−f l o a t ( ’ i n f ’ )
return math . exp(−log_prob_sum / ( len ( t o k e n s )− 2 ) )
4
6.2.2 Obliczanie współczynnika powtórzeń
def r e p e t i t i o n _ r a t e ( t e x t , n =3):
t o k e n s = t e x t . s p l i t ( )
ngrams = [ tuple ( t o k e n s [ i : i+n ] ) f o r i in range ( len ( t o k e n s )−n +1)]
unique_ngrams = set ( ngrams )
return 1− len ( unique_ngrams ) / len ( ngrams )
6.2.3 Analiza rozkładu części mowy
def p o s _ d i s t r i b u t i o n _ d i s t a n c e ( o r i g i n a l _ t e x t s o r i g i n a l _ p o s = [ ]
generated_pos = [ ]
, g e n e r a t e d _ t e x t s ) :
f o r t e x t in o r i g i n a l _ t e x t s :
doc = n l p ( t e x t )
o r i g i n a l _ p o s . e xt en d ( [ toke n . pos_ f o r to ken in doc ] )
f o r t e x t in g e n e r a t e d _ t e x t s :
doc = n l p ( t e x t )
generated_pos . e xt en d ( [ token . pos_ f o r to ken in doc ] )
# O b l i c z o r i g _ d i s t g e n _ d i s t r o z k a d y p r a w d o p o d o b i e s t w a
= c a l c u l a t e _ d i s t r i b u t i o n ( o r i g i n a l _ p o s )
= c a l c u l a t e _ d i s t r i b u t i o n ( generated_pos )
return j e n s e n s h a n n o n ( o r i g _ d i s t , g e n _ d i s t )
6.3 Ankieta human-centriczna
6.3.1 Kwestionariusz oceny jakości
1. Płynność (1-5): Czy tekst jest gramatycznie poprawny?
2. Spójność (1-5): Czy zdania logicznie wynikają z siebie?
3. Sensowność (1-5): Czy tekst ma logiczny sens?
4. Zgodność stylistyczna (1-5): Czy tekst pasuje do korpusu źródło-
wego?
5
6.3.2 Metoda przeprowadzenia ankiety
• 10 oceniających niezależnych
• Po 5 tekstów z każdego modelu (2. i 3. rzędu)
• Teksty prezentowane w losowej kolejności
• Średnia ocen dla każdej kategorii
6.4 Analiza statystyczna wyników
6.4.1 Testy istotności
• Test t-Studenta dla różnic w średnich ocenach
¯
X1−
¯
X2
t=
(8)
sp
2
n
• Test ANOVA dla porównania wielu modeli
• Poziom istotności α= 0.05
6.5 Wizualizacja wyników
6.5.1 Wykresy do przygotowania
• Wykres słupkowy porównujący metryki dla 2. i 3. rzędu
• Box plot rozkładów ocen human-centricznych
• Heatmapa macierzy przejść dla najczęstszych sekwencji
• Wykres rozrzutu długości zdań vs. współczynnik powtórzeń
6.6 Interpretacja wyników
6.6.1 Kryteria sukcesu
• Perpleksja < 100 dla modelu 3. rzędu
• Współczynnik powtórzeń < 15%
• Średnia ocena human-centriczna > 3.5/5.0
• Istotna statystycznie różnica na korzyść modelu wyższego rzędu
6
6.6.2 Wnioski jakościowe
• Czy model 3. rzędu generuje bardziej spójne teksty?
• Czy wyższy rząd zmniejsza liczbę powtórzeń?
• Jaki jest kompromis między płynnością a kreatywnością?
• Który model lepiej oddaje charakter korpusu źródłowego?
