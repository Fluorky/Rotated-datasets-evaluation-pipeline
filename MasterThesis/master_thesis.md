---
header-includes:
  - \usepackage{graphicx}
  - \usepackage{multicol}
  - \usepackage{ragged2e}
  - \usepackage{tocloft}
  - \renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}

---


\begin{titlepage}
\centering


\includegraphics[width=0.7 \textwidth]{media/ul_logo.png}

\vspace{1cm}

{\LARGE \textbf{Maciej Bujalski}} \\[1cm]

\RaggedRight
{\large
\textbf{Kierunek:} informatyka\\
\textbf{Specjalność:} informatyka stosowana\\
\textbf{Specjalizacja:} aplikacje mobilne\\
\textbf{Numer albumu:} 386012\\
}

\vspace{2.5cm}

\centering
{\Large \textbf{Rotacyjnie inwariantne sieci neuronowe}} \\[2cm]

\begin{flushright}
\large
\textbf{Praca magisterska} \\
wykonana pod kierunkiem \\
dr Krzysztofa Podlaskiego \\
w Katedrze Systemów Inteligentnych, WFiIS UŁ
\end{flushright}

\vfill

{\large Łódź 2025}

\end{titlepage}

\newpage

\tableofcontents

\newpage

# Wstęp

Obrazy otaczają nas z każdej strony, od zdjęć ze smartfonów, zdjęcia
satelitarne, przez monitoring miejski, katalogi produktów i systemy
kontroli jakości na liniach produkcyjnych, po systemy wspomagania jazdy.
Choć współczesne modele rozpoznawania obrazu radzą sobie bardzo dobrze,
w praktyce bywają wrażliwe na pozornie drobne zmiany takie jak
obrócenie obiektu o kilkanaście stopni czy niewielki przechył kamery.
To, co dla człowieka jest naturalne i natychmiast rozpoznawalne (znak
drogowy pod kątem, cyfra obrócona na kartce), dla klasycznej
konwolucyjnej sieci neuronowej bywa problemem. Największy problem to brak
naturalnej inwariantności względem rotacji, standardowe CNN-y „z
definicji” lepiej radzą sobie z przesunięciami niż z obrotami
[@goodfellow2016deep; @dumoulin2016guide].

W ostatnich latach pojawiło się kilka dróg domknięcia tej luki. Jedna to
poszerzanie danych o zrotowane przykłady, które poprawiają odporność, ale
wydłużają trening i nie gwarantują uogólnienia na wszystkie kąty. Druga to
architektury z wbudowaną geometrią: sieci grupowo równoważne (G-CNN,
E(2)-equivariant) [@cohen2016group; @kim2020cycnn], sieci cykliczne
(CyCNN, a w szczególności **CyVGG** i **CyResNet**) operujące na wielu
orientacjach oraz przekształcenia do układów polarnych (linear-polar i
log-polar), które „prostują” rotacje do przesunięć. Cel jest wspólny,
mianowicie by model rozpoznawał „to samo” niezależnie od orientacji,
bez agresywnego dublowania danych.

Niniejsza praca skupia się na praktycznej weryfikacji tych podejść.
Przygotowano zbiory obejmujące m.in. odręcznie napisane cyfry, znaki
drogowe (w kolorze i w odcieniach szarości) oraz syntetyczne obiekty 3D
rzutowane na 2D (klocki LEGO), a następnie rozszerzono je o kontrolowane
rotacje. Zaimplementowano i porównano wybrane architektury rotacyjnie
inwariantne i ich warianty bazowe w **PyTorchu** [@paszke2019pytorch],
mierząc wpływ transformacji (linear-polar vs. log-polar), wyboru
architektury i zakresu kątów na jakość predykcji. Obliczenia realizowano
na kartach graficznych: **NVIDIA GeForce RTX 3070 Ti 8 GB** oraz **RTX 3060 12 GB**,
co skróciło czas trenowania i umożliwiło szeroki przegląd eksperymentów.
Środowisko uruchomieniowe ustandaryzowano z użyciem **Dockera** dla
powtarzalności.

Celem pracy jest nie tylko pokazanie, że „da się” uzyskać odporność na
rotacje, ale przede wszystkim wskazanie, **kiedy** i **jakim kosztem**
ją osiągamy oraz które techniki przynoszą największy zysk względem
klasycznych CNN-ów, ich wpływ na stabilność i szybkość uczenia, a
także które konfiguracje są najpraktyczniejsze w realnych
zastosowaniach (OCR, rozpoznawanie znaków, analiza obiektów technicznych).
W dalszej części pracy przedstawiono podstawy, dane i augmentację,
architektury, środowisko eksperymentalne, protokoły ewaluacji oraz
wyniki z analizą i wnioskami.

\newpage

## Cel i motywacja pracy

Konwolucyjne sieci neuronowe (CNN) charakteryzują się zdolnością do
analizy obrazów z zachowaniem niezmieności względem translacji.
Jednak wciąż brakuje powszechnie uznanych architektur, które
zapewniałyby inwariantność względem rotacji. Celem niniejszej pracy
jest analiza skuteczności rozwiązań zaproponowanych w literaturze,
ukierunkowanych na odporność modeli na rotację danych wejściowych
(np. CyCNN [@kim2020cycnn]).

W ramach pracy zostały przygotowane wzbogacone zbiory danych,
obejmujące m.in. zdjęcia odręcznie napisanych liter, znaki drogowe
(zarówno w kolorze, jak i w odcieniach szarości) oraz obiekty 3D
rzutowane na 2D (klocki LEGO), rozszerzone o kontrolowane rotacje
obrazów.

Na podstawie tych zbiorów została przeprowadzona implementacja i
ewaluacja wybranych architektur rotacyjnie inwariantnych z wykorzystaniem
**PyTorch**. Obliczenia zostały wykonane przy użyciu kart graficznych
NVIDIA, co umożliwiło przyspieszenie trenowania i testowania. Otrzymane wyniki zostały
porównane z rezultatami klasycznych sieci konwolucyjnych w celu oceny
realnych korzyści z użycia rozwiązań inwariantnych do zbiorów z rotacją.

## Opis pracy

Praca magisterska wykorzystuje zaawansowane technologie i narzędzia
wspierające badania nad rotacyjnie inwariantnymi sieciami neuronowymi
oraz ich zastosowaniem w przetwarzaniu obrazów. W realizacji projektu
zastosowano następujące rozwiązania technologiczne:

- **Język programowania Python** - podstawowe narzędzie do implementacji
  algorytmów oraz obsługi frameworków uczenia maszynowego, dzięki
  wszechstronności i bogatemu ekosystemowi bibliotek [@python-docs]
  Środowiska uruchomieniowe były izolowane dzięki użyciu `venv`.

  **Frameworki uczenia maszynowego:**
  - **PyTorch** - elastyczny framework do budowy, trenowania i wdrażania modeli
  ML/DL (w tym własnych warstw, takich jak `CyConv`) [@pytorch-docs].
  - **Optuna** - biblioteka do automatycznej optymalizacji hiperparametrów
  (*HPO*) z obsługą efektywnych strategii próbkowania (**TPE**) oraz
  mechanizmów wczesnego przerywania treningów (**MedianPruner** itp.).
  Umożliwia ona definiowanie przestrzeni poszukiwań, rejestrowanie metryk,
  zapisywanie wyników (np. do CSV/JSON) oraz łatwe odtwarzanie najlepszych
  konfiguracji w postaci *study*. Integracja z PyTorchem odbywa się bez
  zmian w architekturze modeli i pozwala skrócić czas eksperymentów bez
  utraty jakości [@akiba2019optuna].

- **Modele cykliczne (CyCNN).** W pracy zostało przyjęte podejście, w
  którym obraz został przemapowany do współrzędnych $(\rho,\varphi)$.
  Dzięki temu obrót $R_\alpha$ staje się przesunięciem o $\alpha$ po osi
  $\varphi$. Konwolucje zostały zastąpione warstwami cylindrycznymi
  (CyConv) z cyklicznym paddingiem po $\varphi$. Dla każdego filtra
  zostało przygotowanych $n$ orientacji, a odpowiedzi zostały złożone z
  dodatkową osią „orientacja”. Obrót wejścia powoduje cykliczne przesunięcie
  po tej osi, a pooling po orientacjach daje inwariancję względem
  rotacji. Został ustawione takie parametry jak stały środek układu polarnych,
  stała siatka próbkowania oraz biliniarna interpolacja. Padding po
  $\varphi$ został ustawiony na cykliczny. Implementacja została wykonana
  w **PyTorchu** (punkt odniesienia: CyCNN [@kim2020cycnn]).

- **Wykorzystanie akceleracji GPU (NVIDIA)** - obliczenia zostały
  znacząco przyspieszone dzięki użyciu kart **RTX 3070 Ti 8 GB** oraz
  **RTX 3060 12 GB**. Frameworki takie jak PyTorch wspierają **CUDA**, **Tensor** oraz **cuDNN**, co
  umożliwia efektywne wykorzystanie zasobów GPU [@cuda-docs; @cudnn-docs].
  Monitorowanie i diagnostyka zostały wykonane z użyciem narzędzia `nvidia-smi`.

- **Rdzenie CUDA (CUDA Cores).** Podstawowe jednostki wykonawcze
  multiprocesorów strumieniowych (SM) realizują obliczenia arytmetyczne
  w precyzji **FP32/INT32**. Konwolucje oraz mnożenia macierzy są
  wykonywane na rdzeniach CUDA zawsze wtedy, gdy nie są aktywowane
  Tensor Cores (np. czysty FP32 bez **TF32**/AMP). Wydajność zależy od
  obsadzenia SM-ów (occupancy), doboru rozmiaru bloków (wielokrotność
  32 wątków - **warp**), koalescencji dostępu do pamięci globalnej oraz
  wykorzystania pamięci współdzielonej. Warstwa `CyConv2d` wymusza
  `contiguous()` i obecność tensora na CUDA przed wywołaniem jądra;
  duży `workspace` sprzyja kafelkowaniu i ogranicza liczbę odczytów z
  DRAM, co poprawia przepływ danych na SM-ach [@cuda-docs].

- **Tensor Cores (Ampere).** Zastosowane karty graficzne RTX (3070 Ti, 3060) mają
  rdzenie Tensor, które sprzętowo przyspieszają operacje macierzowe
  (konwolucje/matmul). Biblioteki **cuDNN/cuBLAS** na architekturze
  **Ampere** domyślnie mogą używać trybu **TF32** dla obciążeń FP32,
  co daje dodatkowe przyspieszenie bez zmian w modelu.
  Dodatkowo, w **PyTorchu** możliwe jest włączenie **mieszanej precyzji**
  (FP16/BF16) przez **AMP** w miejscach, gdzie to bezpieczne, przy
  włączeniu tego feature, zwykle przyspiesza to trening przy
  porównywalnej jakości (szczegóły znajdują się w dokumentacji).
  [@nvidia_tensorcores; @nvidia_tf32; @micikevicius2018mixed; @pytorch_amp]

- **System operacyjny: Linux (Ubuntu LTS).** Główne środowisko uruchomieniowe stanowił system operacyjny
  **Ubuntu** (dystrybucja LTS) posiadający stabilne
  jądro, pakiety z APT, łatwą integrację ze sterownikami NVIDIA i CUDA.
  Treningi uruchamiane były **lokalnie** na maszynach z GPU NVIDII.
  [@ubuntu_docs]. Dla zgodności ze środowiskami Windows używano też
  wariantu **WSL2** (ten sam obraz Dockera i ta sama konfiguracja)
  [@wsl_docs].

- **Konteneryzacja za pomocą Dockera** - odizolowane środowiska
  uruchomieniowe ułatwiły replikację i współdzielenie projektu, także
  z obsługą GPU [@docker-docs].

\newpage

## Zakres tematyczny

Niniejsza praca dotyczy odporności modeli klasyfikacji obrazów na rotacje
w płaszczyźnie 2d. Skupiono się na porównaniu klasycznych architektur z ich
wersjami rotacyjnie inwariantnymi oraz na wpływie przekształceń polarnych
na jakość predykcji. Badania zostały przeprowadzone na obrazach 2D i ich
rotacjach planarnych. W szczególności zostały porównane warianty bazowe
**VGG/ResNet** z wersjami cyklicznymi **CyVGG/CyResNet**, a także wpływ
mapowania **linear-polar** i **log-polar**. Eksperymenty zostały wykonane
na zbiorach **MNIST**, **GTSRB_gray**, **GTSRB_RGB** i **LEGO** z
kontrolowanymi rotacjami.

### Ujęte w zakresie

- **Architektury modeli:** zostały zaimplementowane i porównane warianty
  bazowe **VGG** oraz **ResNet** [@simonyan2014vgg; @he2016resnet], a także
  ich wersje cykliczne **CyVGG** i **CyResNet** (modele rotacyjnie
  inwariantne) [@kim2020cycnn].

- **Przekształcenia polarne:** została oceniona użyteczność mapowania
  **linear-polar** oraz **log-polar** jako etapów wstępnego
  przetwarzania służących „prostowaniu” rotacji do przesunięć
  [@reddy1996logpolar; @kim2020cycnn].

- **Zbiory danych:** zostały wykorzystane następujące zbiory:
  **MNIST** (odręczne cyfry, 28x28 przeskalowane do 32x32, grayscale) [@lecun1998mnist],
  **LEGO** (syntetyczne obiekty 3D rzutowane na 2D, 96x96, grayscale),
  **GTSRB** (znaki drogowe, 32x32, grayscale) oraz **GTSRB_RGB**
  (wersja kolorowa) [@stallkamp2011gtsrb]. Zbiory zostały rozszerzone
  o kontrolowane rotacje oraz zostały przygotowane spójne podziały zbiorów na
  train/val/test.

- **Augmentacja i protokół nauki, validacji oraz testów:**
  zostały zdefiniowane zakresy kątów do sprawdzenia,
  liczba treningów, podział na zbiory train/val/test (uczący/
  walidacyjny/testowy), z możliwością powtórzenia trenowań.

- **Środowisko i implementacja:** został wykorzystany **PyTorch** z
  akceleracją **CUDA/cuDNN** na kartach **NVIDIA GeForce RTX 3070 Ti 8 GB**
  oraz **NVIDIA GeForce RTX 3060 12 GB** [@pytorch-docs; @cuda-docs;
  @cudnn-docs]. Przygotowane zostały skrypty w
  Pythonie do trenowania, testowania i ewaluacji [@python-docs].
  Środowisko uruchomieniowe zostało ustandaryzowane z
  użyciem **Dockera** [@docker-docs].

- **Metryki i analiza:** została przeprowadzona ocena jakości (accuracy,
  macierze pomyłek), analiza stabilności (średnia/mediana/odchylenie
  standardowe), wpływ kąta rotacji na skuteczność oraz koszt
  obliczeniowy (czas trenowania, rozmiar modelu).

### Poza zakresem
- Detekcja obiektów i segmentacja - w pracy rozpatrywana jest wyłącznie
  klasyfikacja [@he2017mask; @ronneberger2015unet].

- Inwariancja względem skali, ścinania i pełnych przekształceń afinicznych -
  analizowana jest tylko rotacja w płaszczyźnie [@lowe2004sift; @jaderberg2015stn].

- Rotacje w geometrii 3D oraz zagadnienia widzenia stereo - pozostają
  poza zakresem [@esteves2018spherical; @hartley2004mv].

- Trening na bardzo dużych korpusach z pre-treningiem self-supervised oraz
  szerokim AutoML/hyper-search - nie został realizowany
  [@chen2020simclr; @he2020moco; @li2018hyperband].

- Odporność na silne zakłócenia (szum, okluzje) - poza zakresem, gdyż skupiono
  się wyłącznie na rotacji [@hendrycks2019imagenetc; @devries2017cutout].


### Artefakty pracy

- Repozytoria z kodem, skryptami i plikami konfiguracyjnymi.
- Pliki konfiguracyjne eksperymentów i opis danych.
- Wytrenowane wagi modeli (wybrane checkpointy) oraz raporty z ewaluacji.
- Tekst pracy z dokumentacją eksperymentów i wnioskami.

### Organizacja pracy

Struktura pracy została ułożona tak, by od podstaw przejść do wyników.
W rozdziale **Podstawy teoretyczne** zostały zebrane pojęcia i narzędzia:
CNN, inwariancja/ekwawariancja, przekształcenia polarne oraz prace
pokrewne (G-CNN, E(2)-equivariant, CyCNN). W **Opisie zbiorów danych**
zostały przedstawione MNIST, LEGO, **GTSRB_gray** i **GTSRB_RGB** oraz
sposób augmentacji (rotacje, podziały train/val/test). W
**Architekturach modeli** zostały opisane warianty bazowe
**VGG/ResNet** oraz ich wersje cykliczne **CyVGG/CyResNet**, wraz z
transformacjami linear-polar / log-polar. Rozdział **Implementacja i
środowisko** zawiera szczegóły techniczne: **PyTorch**, **CUDA i cuDNN**
(RTX 3070 Ti 8 GB, RTX 3060 12 GB), **Docker**, strukturę projektu i
skrypty. W **Eksperymentach** zostały zdefiniowane scenariusze, metryki
i sposób ewaluacji. Dalej, w **Porównaniu wyników**, zostały zestawione
modele (VGG vs. CyVGG, ResNet vs. CyResNet, wpływ transformacji) i
omówiona stabilność/czas. Na końcu sekcja **Wnioski** zbieraja najważniejsze
**obserwacje** i wskazuje kierunki dalszych badań. Sekcja **Aneks** zawiera kody
i dodatkowe wykresy.

\newpage

# Podstawy teoretyczne

Celem tego rozdziału jest uporządkowanie pojęć, które są potrzebne do
zrozumienia dalszych eksperymentów. Najpierw zostały przedstawione podstawy
klasycznych sieci konwolucyjnych takich jak: idea splotu, lokalne pola recepcyjne,
współdzielenie wag i wynikająca z tego ekwiwariancja względem translacji
[@lecun1998gradient; @goodfellow2016deep; @dumoulin2016guide]. Następnie omawaine są
parametry geometrii warstwy (stride, padding, dylacja), zależności rozmiarów
wejścia i wyjścia oraz sposób, w jaki pooling buduje praktyczną inwariancję na
przesunięcia.

Następnie przedstawiona zostaje różnica między **ekwiwariancją**, a
**inwariancją** oraz pokazane są ograniczenia klasycznych CNN w kontekście
rotacji. Ten brak zgodności grupowej dla obrotów motywuje dwie ścieżki
rozwijane w literaturze, pierwsza  to mapowanie do układu biegunowego i operowanie na osi
kąta w sposób cykliczny (linia **CyCNN**), zaś druga to konstrukcje oparte o sploty
grupowe i jądra sterowalne w grupie $\mathrm{E}(2)$ (sieci **E(2)-equivariant**)
[@bronstein2021gdl; @kim2020cycnn; @cohen2016group]. W tej pracy wykorzystywana
jest pierwsza ścieżka, ponieważ pozwala zachować standardowy pipeline i
porównywalny budżet parametrów, a jednocześnie wprowadzić kontrolowaną
ekwiwariancję rotacyjną, która po agregacji orientacji przechodzi w inwariancję.

Dla kompletności omówione zostaną też praktyczne aspekty przekształceń polarnych
(linear-polar i log-polar), sposób liczenia pól recepcyjnych po takich
mapowaniach oraz wpływ decyzji implementacyjnych (interpolacja, wybór środka,
cykliczny padding po $\varphi$) na stabilność uczenia. Taki zestaw podstaw
pozwala czytelnie oddzielić wpływ **augmentacji** od wpływu **architektury** i
stanowi fundament pod analizę wyników w dalszych rozdziałach.


## Wprowadzenie do sieci konwolucyjnych (CNN)

Sieci konwolucyjne (CNN) zostały zaprojektowane do pracy na danych o strukturze
siatkowej lub macierzowej, takich jak obrazy dwuwymiarowe (2D). Ich kluczowe cechy to
**lokalne receptywne pola**, **współdzielone wagi** oraz **operacja splotu**. Dzięki
temu możliwe jest skalowanie modeli na duże obrazy oraz lepsze uogólnianie niż w
przypadku sieci posiadającej pełne połączenia.

Zamiast analizować cały obraz jednocześnie, CNN wykorzystują mały filtr, który
przesuwa się po lokalnych fragmentach obrazów. W ten sposób uczą się prostych
detektorów (np. krawędzi, tekstur, kształtów), zaś w wyższych warstwach
bardziej złożonych cech [@lecun1998gradient; @goodfellow2016deep].

Dla przesunięcia $\mathcal T_t$ oraz jądra $K$ zachodzi własność
**ekwiwariancji translacyjnej**:
$$
\mathcal T_t(X) * K \;=\; \mathcal T_t\!\big(X * K\big).
$$

Intuicyjnie oznacza to, że jeśli obraz zostanie przesunięty, to odpowiadająca
mu mapa cech również przesunie się o ten sam wektor.
**Inwariancja na przesunięcia** osiągana jest w praktyce poprzez pooling
(lokalny lub globalny) bądź zwiększenie kroku (stride). Rzadsze próbkowanie
powoduje, że wynik klasyfikacji nie jest zależny od dokładnej pozycji obiektu
[@dumoulin2016guide; @goodfellow2016deep].

### Operacja splotu

Intuicyjnie ten sam mały filtr „przesuwa się” po obrazie i oblicza ważoną sumę
pikseli. Dzięki temu uzyskuje się **współdzielenie wag** (liczba parametrów nie
rośnie wraz z $H,W$) oraz **lokalność obliczeń**
[@dumoulin2016guide; @goodfellow2016deep].

**Kształty.**  \
Wejście: $X \in \mathbb{R}^{C_{\text{in}}\times H\times W}$.  \
Zestaw jąder: $K \in \mathbb{R}^{C_{\text{out}}\times C_{\text{in}}\times k\times k}$.  \
Wyjście: $Y \in \mathbb{R}^{C_{\text{out}}\times H'\times W'}$.  \

**Definicja (pojedynczy kanał wyjściowy $c$):**
$$
Y_c(u,v)=\sum_{i=1}^{C_{\text{in}}}\sum_{a,b}
K_{c,i}(a,b)\,X_i(u-a,\;v-b).
$$

W praktyce większość frameworków oblicza **korelację krzyżową** (bez
odwracania jądra), pomimo tego, że w API funkcja nazywana jest `conv` [@dumoulin2016guide].
Nie ma to jednak końcowo znaczenia dla procesu uczenia, bo sieć i tak ostatecznie dobierze
właściwe wagi.

#### Stride, padding, rozmiary  \

Parametry geometrii warstwy:

- **padding** $p$ - ile pikseli dodajemy na brzegach;
- **stride** $s$ - co ile pikseli przesuwamy okno;
- **dylacja** $d$ - „rozciąga” jądro poprzez wstawienie przerw między próbkami.

#### Same/valid/stride, a ekwiwariancji  \

Dokładna ekwiwariancja translacyjna zachodzi przy splocie bez zmiany rozmiaru.
W praktyce **padding „same”**, **stride $>1$** i **pooling** wprowadzają drobne
odchylenia (aliasing na siatce próbkowania), co obniża „idealność”
ekwiwariancji, efekt ten jest znany i opisywany w literaturze
[@dumoulin2016guide; @azulay2019small].

**Rozmiar wyjścia** (dla jądra $k\times k$):
$$
H'=\Big\lfloor \frac{H+2p-d\,(k-1)-1}{s}\Big\rfloor+1,\qquad
W'=\Big\lfloor \frac{W+2p-d\,(k-1)-1}{s}\Big\rfloor+1.
$$

**Typowe ustawienia.**  \
- *valid*: $p=0$ - mapy cech maleją;  \
- *same* (dla $s=1$): $p=\lfloor k/2\rfloor$ - $H'=H$, $W'=W$;  \
- *stride $>1$*: wbudowane **podpróbkowanie** (mniej obliczeń, mniejsza
  rozdzielczość);  \
- *dylacja $>1$*: większe **efektywne** pole widzenia bez nowych parametrów
  (częste w detekcji/segmentacji) [@dumoulin2016guide].  \

### Receptywne pole

Receptywne pole to fragment obrazu „widziany” przez daną aktywację w mapie cech.
Im głębsza warstwa, tym pole jest większe, ponieważ kolejne sploty agregują
informacje z coraz większych fragmentów wejścia. Dla jąder $k_\ell$ i kroków
$s_\ell$ zachodzi:
$$
R_1 = k_1,\qquad
R_\ell = R_{\ell-1} + (k_\ell-1)\!\!\prod_{j<\ell}s_j.
$$

W praktyce nie wszystkie piksele w obrębie tego pola mają taki sam wpływ. Największy
wpływ ma obszar centralny, a znaczenie maleje ku brzegom, przy czym rozkład wpływu
przypomina rozkład Gaussa. Z tego powodu w bazowych architekturach VGG i ResNet dobiera się
głębokość tak, aby przy stosowanej rozdzielczości objąć cały obiekt bez utraty
istotnego kontekstu [@luo2016erf].

W wariantach **CyCNN** po przejściu do współrzędnych $(\rho,\varphi)$ oraz przy
**cyklicznym paddingu** wzdłuż $\varphi$ sieć „widzi” pełny zakres orientacji bez
artefaktów brzegowych, co ułatwia stabilne uczenie cech niezależnych od kąta
[@kim2020cycnn].


#### Receptywne pole w układzie polarnym\

Po mapowaniu $(x,y)\!\to\!(\rho,\varphi)$ receptywne pole staje się „wąskim paskiem”
wzdłuż promienia $\rho$ i przy czym stabilnym po kącie $\varphi$. Dzięki temu obrót
$\mathcal{R}_\alpha$ na wejściu jest równoważny **przesunięciu** o $\alpha$ po osi
$\varphi$. Warstwy typu `CyConv` z **cyklicznym paddingiem** wzdłuż $\varphi$ nie
„tną” informacji na brzegach, co oczywiście wzmacnia ekwiwariancję rotacyjną [@kim2020cycnn].


### Nieliniowości i normalizacja

Blok konwolucyjny pozostaje **taki sam** we wszystkich wariantach (bazowych i
cyklicznych). Celem porównania jest wpływ **rotacji**, a nie dobór aktywacji.

Zastosowano standardową normalizację, celem stabilizacji uczenie. W wariantach
**CyCNN** statystyki normalizacji liczone są **wspólnie po osi orientacji**, tak aby
**nie faworyzować żadnego kąta** i nie naruszać własności rotacyjnych
[@ioffe2015batchnorm; @kim2020cycnn].

### Pooling i część klasyfikacyjna

W modelach **CyCNN** inwariancja względem rotacji uzyskiwana jest przez
**agregację po orientacjach** (pooling po osi kątów). Następnie, we wszystkich
modelach, stosowany jest **global average pooling (GAP)** oraz pojedyncza warstwa
liniowa w klasyfikatorze. GAP redukuje liczbę parametrów i zmniejsza zależność
od położenia w obrębie map cech [@lin2014network]. Część klasyfikacyjna pozostaje
taka sama w wariantach bazowych i cyklicznych, aby izolować wpływ części
„rotacyjnej” [@kim2020cycnn].

#### Pooling po orientacjach - szczegóły praktyczne  \

Agregacja po osi *orientacja* (avg lub max) realizuje **inwariancję rotacyjną**.
Na wynik końcowy wpływa liczba orientacji **n**, czyli większe **n** oznacza dokładniejszą
rozdzielczość kątową (mniejszy błąd zaokrąglenia $2\pi/n$), ale też wyższy koszt
obliczeń i pamięci. W eksperymentach utrzymano identyczny klasyfikator za
poolingiem, aby jednoznacznie mierzyć wpływ części „rotacyjnej”
[@kim2020cycnn].

### Trening

Protokół trenowania został **zamrożony** między wariantami (te same: liczba epok,
rozmiar batcha, budżet obliczeniowy, wczesne zatrzymanie - wsyzstko to zgodnie z
planem eksperymentu). Augmentacje ograniczono do tych **niezależnych od rotacji**
w testach „czysto architektonicznych”. Augmentację rotacją stosowano wyłącznie w
eksperymentach kontrolnych, tak aby pokazać różnicę między „augmentacją” a
„architekturą”.

## Modele Cy oraz ich triki architektoniczne

Jako bazy zastosowano VGG-19 (bloki 3×3) i ResNet-56 w wariancie
CIFAR (bloki 3×3). W wersjach cyklicznych każdą warstwę Conv2d zastąpiono
CyConv2d. Po stronie geometrii zastosowano cykliczny padding wzdłuż osi
kątowej $\varphi$. Układ bloków, liczby kanałów, BN/ReLU, GAP i
klasyfikator zostały pozostawione bez zmian, tak aby utrzymać porównywalny budżet
parametrów/FLOPs względem bazowych modeli. Na poziomie definicji modeli nie
dodano jawnej osi „orientacja” ani osobnego poolingu po orientacjach,
jeśli mechanizmy rotacyjne są użyte, są one enkapsulowane w implementacji
CyConv2d (jądro CUDA), a nie w topologii sieci. Nie zostały prowadzone modyfikacje
niezwiązane z rotacją (np. zmiana funkcji aktywacji, normalizacja, głębokość,
rozmiar jąder, liczby kanałów czy regularizacji), aby nie mieszać
ich wpływu z efektem komponentu rotacyjnego[@kim2020cycnn].

### Ekwiwariancja translacyjna (kontrast do rotacyjnej)

Dla przesunięcia $\mathcal{T}_t$ i splotu zachodzi:
$$
\mathcal{T}_t(X) * K \;=\; \mathcal{T}_t\!\big(X*K\big),
$$
co tłumaczy, dlaczego klasyczne CNN dobrze radzą sobie z translacją
[@dumoulin2016guide]. Brak analogicznego mechanizmu dla rotacji motywuje
użycie przekształceń polarnych i/lub modeli cyklicznych w dalszej części.

#### Ekwiwariancja rotacyjna w dyskretnej grupie $C_n$  \

Dla indeksu orientacji $k\in\{0,\dots,n-1\}$ i kąta
$\theta_k=\tfrac{2\pi k}{n}$ ekwiwariancję można zapisać jako

$$
\Phi\!\big(\mathcal{R}_{\theta_k} X\big)[\cdot,\;m]
\;=\; \Phi(X)[\cdot,\;m\!+\!k \bmod n],
$$

czyli **obrót wejścia = cykliczne przesunięcie po osi orientacji**. Następnie
pooling po tej osi znosi zależność od kąta (inwariancja) [@kim2020cycnn].

**Ekwiwariancja** (przewidywalna zmiana reprezentacji):
$$
\Phi(\mathcal{T}_t X)=\Pi_t\,\Phi(X),\qquad
\Phi(\mathcal{R}_\alpha X)=\Pi_\alpha\,\Phi(X),
$$
gdzie $\Pi$ to działanie grupy na cechach (dla translacji - przesunięcie map,
dla rotacji - np. **cykliczny shift** po osi „orientacja”)
[@goodfellow2016deep; @bronstein2021gdl].

## Inwariancja translacyjna i rotacyjna

Niech $\mathcal{T}_t$ oznacza przesunięcie o wektor $t$, $\mathcal{R}_\alpha$ -
obrót o kąt $\alpha$, a $\Phi$ - mapę cech / model.

**Inwariancja** (wynik nie zależy od transformacji):
$$
\Phi(\mathcal{T}_t X)=\Phi(X),\qquad
\Phi(\mathcal{R}_\alpha X)=\Phi(X).
$$

W praktyce:  \
- **translacja:** uśrednianie / podpróbkowanie (GAP, stride);  \
- **rotacja:** (a) augmentacja o obroty, (b) architektura ze śledzeniem
  orientacji (oś „kąt” + pooling po orientacjach), (c) mapowanie do
  współrzędnych polarnych, gdzie obrót staje się przesunięciem po osi
  $\varphi$ [@dumoulin2016guide; @reddy1996logpolar; @kim2020cycnn].

### Linear-polar vs. log-polar

- **Linear-polar:** równy przyrost w pikselach po $\rho$ i $\varphi$. Stabilne kąty,
  prosta implementacja. Rozwiązanie to jest dobre pod **inwariancję rotacyjną**.
- **Log-polar:** skala rośnie logarytmicznie po $\rho$ - zmiany skali stają się
  przesunięciami po osi promienia. Jest to idealne rozwiązanie pod **rotacje i skale**,
  ale bliżej środka rośnie gęstość próbkowania i przy tym wrażliwość na wybór środka [@reddy1996logpolar].

W praktyce stosowana jest interpolacja biliniarna wraz z **cyklicznym paddingiem po $\varphi$**, oraz stałym
środkiem. Blisko $\rho{=}0$ warto wygładzić / wykluczyć kilka próbek, aby uniknąć „osobliwości” środka [@kim2020cycnn].

\newpage

## Problemy z rotacyjną inwariancją w klasycznych CNN

W praktyce klasyczne CNN dobrze znoszą przesunięcia, ale nie domykają
symetrii obrotu. Wynika to z natury splotu na dyskretnej siatce, efektów
interpolacji oraz braku jawnej reprezentacji kąta w strumieniu cech.
Poniżej zebrane zostały najważniejsze źródła nieinwariancji, które bezpośrednio
wpływają na wyniki i ich interpretację:

- **Kierunkowość filtrów.** Małe jądra `3×3` i `5×5` reagują głównie na
  jedną orientację. Aby pokryć wiele kątów, sieć musiałaby nauczyć się
  wielu obróconych kopii tych samych detektorów, co zwiększa zapotrzebowanie
  na dane i parametry. Kompozycja kilku warstw częściowo pomaga, ale bez
  mechanizmów ukierunkowanych na kąt problem nie znika.

- **Augmentacja nie domyka całości.** Obracanie wzbogaca dane, ale pokrywa
  tylko zbiór dyskretnych kątów. Między tymi wartościami pozostaje
  „szczelina” generalizacji, zwłaszcza przy rzadkiej siatce kątów i
  ograniczonym budżecie. Dodatkowo augmentacja wydłuża trening i wnosi
  wariancję związaną z losowym próbkowaniem kątów.

- **Aliasing i interpolacja.** Obrót danego rastra wymaga resamplingu i doboru
  jądra interpolacji. Pojawia się wtedy rozmycie lub aliasing, a wysokie
  częstotliwości są tłumione inaczej zależnie od kąta oraz implementacji
  [@azulay2019small]. Skutkiem jest niespójność odpowiedzi nawet przy
  niewielkich obrotach tego samego obiektu.

- **Krawędzie i padding.** Dopełnianie „same/zero” łamie symetrię przy
  brzegach. W pobliżu krawędzi zmienia się kontekst, więc odpowiedzi nie są
  idealnie ekwiwariantne. Stride i pooling pogłębiają ten efekt przez rzadsze
  próbkowanie i aliasing, co dodatkowo obniża stabilność na małe obroty
  [@dumoulin2016guide].

- **Brak osi orientacji.** W typowych CNN nie zapisuje się jawnie informacji
  o kącie wykrytej cechy. Orientacje „mieszają się” w kanałach, więc późniejsze
  domknięcie do inwariancji (np. przez pooling) nie ma do czego się odnieść.
  Stąd potrzeba osi „orientacja” i operacji cyklicznych lub mapowania do układu
  polarnego.

- **Brak zgodności grupowej.** Standardowy splot gwarantuje ekwiwariancję dla
  translacji, ale nie dla rotacji. Na siatce pikseli obrót nie komutuje
  ze splotem jak przesunięcie. Klasyczne CNN nie mają więc gwarancji, że
  $\Phi(\mathcal R_\alpha X)$ jest prostą transformacją $\Phi(X)$.

- **Wczesne warstwy i pole widzenia.** We wczesnych warstwach receptywne pole
  jest małe, przez co lokalne rotacje bywają nierozróżnialne od innych zmian.
  Szerszy kontekst pojawia się dopiero po poolingach i podpróbkowaniu, co
  jednocześnie obniża precyzję kątową.

- **Interakcja z rozmiarem i kształtem obiektu.** Rotacja zmienia relacje
  między detalami, a siatką próbkowania (np. inna liczba „przecinanych”
  pikseli wzdłuż krawędzi przy różnych kątach). Skutkiem są fluktuacje
  aktywacji i decyzji zależne od rasteryzacji i kąta, a nie od samej klasy.

\newpage

# Przegląd literatury (E(2) equivariant, CyCNN)

Literatura o sieciach ekwiwariantnych rozwija się zasadniczo w dwóch
kierunkach. Pierwszy nurt to podejścia o charakterze geometrycznym,
które mapują obraz do współrzędnych biegunowych i traktują oś kąta jako
wymiar cykliczny. Drugi nurt to modele o ściśle zdefiniowanej
ekwiwariancji względem grupy przekształceń E(2), budowane są one poprzez sploty
grupowe oraz jądra sterowalne projektowane zgodnie z reprezentacjami
grupy. Oba podejścia dążą do reprezentacji odpornej na obrót. Różnią
się jednak stopniem formalizacji, kosztem obliczeniowym i wysiłkiem
inżynierskim potrzebnym do integracji z typowymi pipelinami.


## CyCNN
W rodzinie CYCNN obraz przemapowywany jest do układu
$(\rho,\varphi)$ tak, aby obrót w płaszczyźnie stał się przesunięciem po
osi $\varphi$. Przetwarzanie odbywa się warstwami cylindrycznymi
z dopełnieniem cyklicznym wzdłuż osi danego kąta. Dla każdego filtra przyjmuje
się dyskretny zbiór orientacji opisany grupą $C_n$. Obrót wejścia z
tego zbioru przekłada się na cykliczne przesunięcie po wymiarze
orientacji w mapach cech. Agregacja po orientacjach domyka inwariancję.
Wybór liczby orientacji równoważy rozdzielczość kątową i koszt obliczeń.
Istotne są także decyzje implementacyjne takie jak wybór środka,
interpolacja przy odwzorowaniu oraz właściwe domknięcie brzegu dla
$\varphi=0$ i $\varphi=2\pi$, tak aby uniknąć artefaktów. Zaletą CyCNN jest
zgodność z klasyczną praktyką tworzenia modeli. Zamiana klasycznej konwolucji na
odpowiednik cylindryczny odbywa się bez zmiany interfejsu warstwy, co
ułatwia kontrolowane porównania z wersjami bazowymi przy podobnym
budżecie parametrów i zapotrzebowaniu na moc obliczeniową (FLOPs) [@kim2020cycnn].

## E(2) equivariant i sieci sterowalne
Druga linia prac zaś nie zmienia
układu współrzędnych, lecz definiuje splot bezpośrednio na grupie E(2)
albo na przestrzeniach jednorodnych tej grupy. Filtry konstruowane są
w zgodzie z reprezentacjami, co pozwala dzielić wagi pomiędzy
orientacje oraz zapewniać ekwiwariancję także dla kątów traktowanych
jako wielkości ciągłe. W literaturze opisano zarówno wersje dyskretne
w stylu G-CNN, jak i konstrukcje sterowalne, w których filtry rozwija się
w bazach harmonicznych i ogranicza regułami reprezentacji
[@cohen2016group; @weiler2019general; @cohen2019homogeneous]. Rozszerzenia
obejmują również grupy dihedralne $D_n$, które pozwalają modelować
rotacje i odbicia, a także modele na przestrzeniach jednorodnych, co
ułatwia precyzyjne wskazanie, gdzie ma zajść ekwiwariancja, a gdzie
inwariancja.

## Aspekty implementacyjne i koszt
Modele oparte na formalizmie E(2)zapewniają ścisłe gwarancje 
wynikające z algebry grupy oraz konstrukcji jąder. 
Osiąga się to kosztem większych wymagań obliczeniowych i
pamięciowych oraz bardziej złożonej implementacji. Pojawia się konieczność
definiowania typów pól cech, respektowania ograniczeń na kształt filtrów
i pracy w bazach harmonicznych. Linia CyCNN jest lżejsza wdrożeniowo, ponieważ
wystarcza zastąpić standardowy operator splotu jego wariantem cylindrycznym
i traktować wymiar kąta jako cykliczny. Dokładność ekwiwariancji zależy tu
od liczby rozpatrywanych orientacji, jakości interpolacji oraz stabilnego
wyboru środka. W zamian zachowana jest zgodność z istniejącymi modelami
VGG i ResNet oraz ze standardowymi komponentami, takimi jak BatchNorm,
dropout i GAP.

## Wnioski w kontekście pracy 
Wybrana została architektura z rodziny CyCNN, ponieważ 
ułatwia porównanie z modelami bazowymi i pozwala kontrolować
informację o orientacji bez ingerencji w pozostałe elementy sieci. Mapowanie
do $(\rho,\varphi)$ oraz cykliczne traktowanie osi kąta umożliwiają
zrealizowanie ekwiwariancji na etapie ekstrakcji cech, a agregacja po
orientacjach zamienia ją w inwariancję. Taki układ sprzyja rzetelnemu
porównaniu augmentacji rotacją z architekturą z wbudowaną obsługą rotacji
przy tym samym klasyfikatorze i zbliżonym budżecie parametrów modeli
[@kim2020cycnn; @cohen2016group; @weiler2019general; @cohen2019homogeneous].


\newpage

# Opis zbiorów danych
W pracy wykorzystano cztery zbiory: MNIST, GTSRB (w dwóch wariantach: Gray i
RGB) oraz LEGO. Wszystkie dane zostały ujednolicone pod
kątem rozdzielczości i kanałów oraz znormalizowane per kanał. MNIST
przeskalowano do obrazów **32×32** w skali szarości o 10 klasach [@lecun1998gradient], zaś
GTSRB Gray do obrzów **32×32** w skali szarości mający 43 klasy, 
a GTSRB RGB również przeskalowano do rozdzielczości **32×32** z trzema kanałami 
(też 43 klasy jak w przypadku GTSRB Gray), z zachowaniem oficjalnego podziału na trening i
test (IJCNN 2011) [@stallkamp2011gtsrb; @gtsrb_site]. Zbiór LEGO przygotowano jako
obrazy **96×96** w skali szarości posiadający 50 klas [@hazelzet_lego_kaggle].
Część walidacyjną zbiorów wydzielono z części treningowej we wszystkich zbiorach,
pozostawiając test jak w oryginale. Ujednolicenie rozmiaru wejścia i części klasyfikacyjnej
pozwala porównywać modele bazowe i cykliczne przy tym samym budżecie obliczeń. Szczegóły
formatów (IDX/NPY), normalizacji oraz scenariuszy rotacyjnych zostaly opisane
rozdziałach poświęconych augmentacji i implementacji.

## MNIST (cyfry odręczne)

Zbiór MNIST to klasyczny benchmark rozpoznawania cyfr 0-9
[@lecun1998gradient]. Obejmuje **60 000** próbek uczących i **10 000**
testowych. Obrazy mają rozdzielczość **28×28**, są w skali szarości, a
wartości pikseli mieszczą się w zakresie [0, 255]. W eksperymentach
wartości te są najpierw skalowane do [0, 1], a następnie standaryzowane
per kanał. Szczegóły formatu i struktury plików są dostępne na stronie
projektu [@mnist_web].

Na potrzeby porównań obrazy zostały **przeskalowane do 32×32**, aby
dopasować je do ustawień „cifarowych” stosowanych w VGG i ResNet. Wejście
ma **1 kanał**, a liczba klas wynosi **10**. Normalizacja jest liczona na
zbiorze uczącym, przy czym w praktyce często przyjmuje się wartości z przykładów
referencyjnych PyTorcha: średnia ≈ 0.1307 i odchylenie standardowe ≈
0.3081 [@pytorch]. Podział na zbiory utrzymuje spójność z resztą
eksperymentów: z części treningowej wydzielany jest zbiór walidacyjny
(5 000 próbek), a test pozostaje jak w oryginale.

Wybór MNIST wynika z jego prostoty i „czystości”, co pozwala szybko
iterować i w kontrolowany sposób badać wpływ **rotacji cyfr**. Zbiór
dobrze nadaje się do uczciwego porównania modeli bazowych (VGG/ResNet) z
**wersjami cyklicznymi** (CyVGG/CyResNet) przy tym samym budżecie
obliczeń. Rotacje ujawniają też naturalne przypadki brzegowe, np. pary
**6/9** czy **2/5**, które przy większych kątach bywają mylone,
pozwala to wyraźniej odróżnić wpływ **augmentacji** od wpływu **architektury**.

W części poświęconej augmentacji wprowadzane są kontrolowane scenariusze
kątowe: wariant **bez rotacji** jako punkt odniesienia, warianty z
**małymi i średnimi obrotami**, a także **pełny zakres 0-360°**. Celem
jest wykazanie, kiedy **architektura cykliczna** zapewnia przewagę nad
samą augmentacją rotacją.


## GTSRB Gray (znaki drogowe w odcieniach szarości)

**German Traffic Sign Recognition Benchmark (GTSRB)** to zestaw znaków drogowych
z rzeczywistych nagrań, obejmujący **43 klasy**, z oficjalnym podziałem na część
uczącą i testową (IJCNN 2011) [@stallkamp2011gtsrb; @gtsrb_site]. W literaturze
często przytaczana jest również analiza „man vs. computer” z metrykami
porównawczymi [@stallkamp2012manvscomputer].

W wariancie **Gray** zastosowanym w tej pracy wszystkie obrazy zostały
**przeskalowane do 32×32** i **skonwertowane do skali szarości** (1 kanał), tak
aby dopasować je do ustawień „cifarowych” oraz wyizolować wpływ **rotacji** od
informacji barwnej. Zachowano **43 klasy**; walidację wydzielono z **oficjalnej**
części treningowej (spójnie z pozostałymi zbiorami). Zastosowano **normalizację
per-kanał** wyliczaną na zbiorze uczącym.

Wybór wersji w odcieniach szarości motywowany jest tym, że kolor bywa silną
wskazówką (np. czerwone obramowania, niebieskie tła), podczas gdy celem jest
tu głównie **geometria** i ocena, co daje **architektura rotacyjnie inwariantna**
na tle bazowej, bez „pomocy” informacji barwnej. Taki wariant ułatwia też
czyste porównania z **GTSRB RGB** (sekcja poniżej), w których różnice można
przypisać właśnie dostępności koloru.

Zbiór GTSRB stawia kilka typowych wyzwań: nierównomierny rozkład klas, duża
zmienność skali i oświetlenia, efekty perspektywy oraz rozmycie w ruchu. Te
czynniki utrudniają proste uogólnianie i dobrze testują **stabilność względem
rotacji** [@stallkamp2011gtsrb; @stallkamp2012manvscomputer].

W eksperymentach wykorzystano scenariusze kątowe opisane w rozdziale
*Augmentacja i protokół*: wariant **bez rotacji** (baseline), zestawy
**małych/średnich obrotów**, połączenia **różnych** kombinacji kątów oraz
**pełen zakres 0-360°**. Pozwala to porównać **VGG/ResNet** z
**CyVGG/CyResNet** przy identycznym budżecie obliczeń.

\newpage

## GTSRB RGB (znaki drogowe w kolorze)

German Traffic Sign Recognition Benchmark (GTSRB) w wersji kolorowej to ten sam
zestaw 43 klas z oficjalnym podziałem na trening i test
[@stallkamp2011gtsrb; @gtsrb_site]. Na potrzeby eksperymentów obrazy są
przeskalowane do 32×32 (ustawienia „cifarowe”) z zachowaniem trzech kanałów
(RGB). Normalizacja wykonywana jest per kanał na zbiorze uczącym, a walidację
wydzielono z części treningowej analogicznie jak dla wariantu Gray
[@stallkamp2012manvscomputer].

Wersja RGB została włączona, aby ocenić, w jakim stopniu informacja barwna może
kompensować trudność związaną z rotacjami oraz na ile architektury rotacyjnie
inwariantne (CyVGG/CyResNet) nadal poprawiają wyniki względem bazowych modeli
(VGG/ResNet). Zastosowanie tych samych rozmiarów wejścia, tych samych podziałów
oraz tego samego klasyfikatora pozwala na porównanie RGB i Gray w układzie 1:1.

W praktyce kolor stanowi silny sygnał (np. czerwone obramowania zakazów, żółte
trójkąty ostrzegawcze, niebieskie nakazy), lecz nie eliminuje problemów
wynikających z dużej zmienności punktu widzenia, skali, oświetlenia i rozmycia
ruchu. Rotacje pozostają istotnym czynnikiem trudności, a informacja barwna
pomaga głównie odróżniać klasy o podobnych kształtach.

W części eksperymentalnej stosowane są te same scenariusze kątowe co wcześniej:
wariant bez rotacji jako punkt odniesienia, warianty z małymi i średnimi
obrotami, połączenia różnych kombinacji kątów oraz pełny zakres 0-360°. Dzięki
temu zachowana jest porównywalność między VGG/ResNet a CyVGG/CyResNet przy
jednakowym budżecie obliczeń.


## LEGO (obiekty 3D rzutowane na 2D)

Zbiór **Images of LEGO Bricks** (Kaggle) [@hazelzet_lego_kaggle] obejmuje obrazy
elementów LEGO renderowanych jako rzuty 2D. W tej pracy obrazy zostały
skonwertowane do skali szarości i przeskalowane do **96×96**, aby zachować
detale klocków. Ustalono **50 klas** (wejście 1-kanałowe), walidację
wydzielono z części treningowej analogicznie jak w pozostałych zbiorach, a
normalizacja jest liczona per kanał na zbiorze uczącym.

Wybór zbioru LEGO motywowany jest tym, że obiekty mają złożone kształty i
drobne szczegóły, co stanowi naturalny test wrażliwości na orientację. W
odróżnieniu od MNIST (proste cyfry) i GTSRB (silny sygnał koloru), LEGO lepiej
izoluje **geometrię** obiektu, czyli układ wypustek i światłocień, dzięki czemu
różnice między podejściem augmentacyjnym a architektonicznym
(**CyVGG/CyResNet** vs **VGG/ResNet**) są czytelniejsze.

W eksperymentach zastosowano te same scenariusze kątowe co w innych zbiorach:
wariant bez rotacji jako punkt odniesienia, warianty z małymi i średnimi
obrotami, połączenia różnych kombinacji kątów oraz pełny zakres **0-360°**.
Porównania są prowadzone przy tej samej części klasyfikacyjnej i tym samym
budżecie obliczeń, aby izolować wpływ komponentu rotacyjnego.

Przy przekształceniach log-polarnych i niewielkiej rozdzielczości rośnie
gęstość próbkowania w pobliżu środka. W przetwarzaniu wstępnym stosowana jest
interpolacja biliniarna i stały środek układu, co ogranicza artefakty i
utrzymuje porównywalność między wariantami.

\newpage

## Sposób augmentacji danych: zakresy rotacji, łączenie zbiorów

Pipeline obsługuje dwa formaty wejścia. Pierwszy to klasyczny format IDX
(ubyte), stosowany m.in. w zbiorze MNIST. Drugi to tryb NPY, w którym dane
zapisywane są jako `train_images.npy` i `train_labels.npy`, zaś dla części
testowej jako `test_images.npy` i `test_labels.npy`. Niezależnie od formatu
zastosowana jest ta sama logika budowania zbiorów danych oraz ich podziału na
`train` i `test`. Dodatkowo, w ścieżce IDX dla MNIST nazwa pliku `t10k` jest
zamieniana na `test` przed uruchomieniem rotacji.


### Rotacje

Augmentacja rotacją występuje w dwóch wersjach. W pierwszej stosowane są kąty
stałe: dla każdej z góry zadanej wartości tworzony jest osobny zestaw nazwany
według szablonu `rotated-{theta}`. W praktyce wykorzystywane są dwie siatki
kątów: co 30° (30, 60, …, 330) oraz co 45° (45, 90, …, 315). Dostępny jest
również preset łączny `fixed_all`, który obejmuje te obie siatki. Dla każdej
wartości kąta przygotowywane są oddzielne zbiory treningowe i testowe.

Druga wersja opiera się na przedziałach kątów. Tworzone są zbiory
`rotated-a-b` dla dwunastu zakresów: [0,30), [30,60), …, [330,360).
Każdej próbce przypisywany jest losowy kąt z rozkładu jednostajnego w ramach
danego przedziału. Losowanie odbywa się niezależnie dla każdej próbki i
każdego przedziału, co zwiększa różnorodność danych.

Parametry przekształceń są stałe w obrębie formatu. W trybie NPY obrót
wykonywany jest wokół środka kadru, z interpolacją liniową, bez
rozszerzania płótna, a piksele wypadające poza obraz wypełniane są stałym
kolorem tła. W trybie IDX używana jest funkcja `PIL.Image.rotate` w
ustawieniach domyślnych, co utrzymuje stały rozmiar wyjściowy.

### Łączenie zbiorów

Na podstawie zbiorów obrotowych tworzone są zbiory połączone. Zapisywane są
one w folderze `merged_datasets/`, a ich nazwy zaczynają się od prefiksu
`merged_`. Dla kątów stałych powstają zestawy `merged_fixed_30`,
`merged_fixed_45` oraz `merged_fixed_all`. Dla wariantu przedziałowego
dostępne są m.in. `merged_range_0_90`, `merged_range_90_180`,
`merged_range_180_270`, `merged_range_270_360`, a także szersze
`merged_range_0_180`, `merged_range_180_360` oraz pełny
`merged_range_full_0_360`. Każdy z tych presetów może być rozszerzany o zbiór
bez rotacji, co oznaczane jest dopiskiem `_plus_non_rotated`. Dla każdego
presetu przygotowywane są osobno zbiory `train` i `test`.

Sposób łączenia zależy od formatu. W IDX pliki `*-images-idx3-ubyte` i
`*-labels-idx1-ubyte` są sklejane, a nagłówki aktualizowane są o nową liczbę
próbek. W NPY wykonywana jest konkatenacja macierzy obrazów i wektorów
etykiet wzdłuż osi próbek.

### Organizacja katalogów

W katalogu bazowym znajduje się folder zbioru źródłowego, np. `dataset_X`, z
plikami `train_images.npy`, `train_labels.npy`, `test_images.npy`,
`test_labels.npy`. Obok tworzone są katalogi wariantów obrotowych, takie jak
`rotated-30` czy `rotated-0-30`, z analogicznymi plikami dla podziałów
`train` i `test`. Zbiory połączone zapisywane są w `merged_datasets/`, m.in.
w `merged_fixed_30`, `merged_range_180_360_plus_non_rotated` oraz
`merged_range_full_0_360_plus_non_rotated`, również z kompletami plików
treningowych i testowych.

### Scenariusze trenowanie - test

Do porównań wykorzystywany jest plik JSON z opisem scenariuszy ewaluacji.
Nazwy w tym pliku odpowiadają ścieżkom na dysku. Przykładowe klucze
(wartości mają tę samą postać) to: 
- `dataset_LEGO_non_rotated`,   
- `merged_datasets/merged_fixed_30`,  
- `merged_datasets/merged_fixed_30_plus_non_rotated`,   
- `merged_datasets/merged_range_0_180`,  
- `merged_datasets/merged_range_0_180_plus_non_rotated`,  
- `merged_datasets/merged_range_180_360_plus_non_rotated`,  
- `merged_datasets/merged_range_full_0_360_plus_non_rotated`,  
- `rotated-30`,     
- `rotated-45`,  
- `rotated-0-30`,   
- `rotated-90-120`.  
Dla każdego zbioru treningowego przypisywana jest lista zbiorów testowych. Zawsze
uwzględniany jest zbiór bazowy bez rotacji, sam zbiór treningowy oraz
dodatkowe zbiory rotowane dobrane zgodnie z ustalonym limitem.

\newpage

# Architektury modeli (VGG-E, ResNet-56, CyCNN)

W pracy wykorzystano bazowe architektury **VGG** (wariant **E**) i
**ResNet** (wariant **56**) w ustawieniu dla **CIFAR** (obrazy 32×32). Warstwy
splotowe to głównie `3×3` z `padding=1`. W VGG po każdym bloku stosowany jest
`MaxPool2d(2)`, a w ResNecie rozdzielczość zmniejszana jest przez `stride=2`.
Po części splotowej występuje **global average pooling (GAP)** oraz prosty
**klasyfikator**. W VGG używana jest wersja z normalizacją (VGG\_bn) oraz
dwustopniowy klasyfikator `512→512→C` z dropoutem (w implementacji`AdaptiveAvgPool2d((1,1))` + `nn.Sequential`
z dwiema warstwami liniowymi i wyjściem `C`).

## VGG - wariant E (VGG-19, „3×3 everywhere”)

Układ zgodny z [@simonyan2014vgg], dostosowany do 32×32 (CIFAR). W plikach
**VGG** i **CyVGG** konfiguracja **E** to:
`[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
'M', 512, 512, 512, 512, 'M']`. Warstwy budowane są przez funkcję
`make_layers(cfg['E'], ...)`, z opcją BatchNorm (`*_bn`) i z `MaxPool2d`
wstawianym w miejscach oznaczonych symbolem `'M'`. Za częścią splotową
stosowane jest `AdaptiveAvgPool2d((1,1))`, następnie spłaszczenie (flatten) i dwie
warstwy liniowe `512→512→C` z dropoutem.

- **Blok 1:** 64, 64 → MaxPool (32→16)
- **Blok 2:** 128, 128 → MaxPool (16→8)
- **Blok 3:** 256×4 → MaxPool (8→4)
- **Blok 4:** 512×4 → MaxPool (4→2)
- **Blok 5:** 512×4 → MaxPool (2→1)

W **CyVGG** każdą `Conv2d` zastąpiono **`CyConv2d`** (jest interfejs zgodny
z `Conv2d`: `3×3`, `padding=1`, ze wsparciem dylacji ang. `stride`). Klasyfikator i
układ bloków pozostają takie same jak w przypadku VGG.


## ResNet-56 (CIFAR, 6n+2 z n=9)

Schemat jak w [@he2016resnet] dla datasetu CIFAR. Na wejściu `Conv 3×3`, a dalej **3
grupy** po **n=9** bloków `BasicBlock`. Grupa 1 (16 kanałów, `stride 1`),
Grupa 2 (32 kanały, pierwszy blok `stride 2`), Grupa 3 (64 kanały, pierwszy
blok `stride 2`). `BasicBlock` ma postać: `Conv3×3 → BN → ReLU → Conv3×3 → BN`
+ skrót.

Downsample - **opcja A** (CIFAR): podpróbkowanie przestrzenne `x[:, :, ::2, ::2]`
i dopełnienie kanałów przez `F.pad(...)`. Zakończenie: **GAP** z warstwą liniowa `64→C`.

W **CyResNet** wszystkie `nn.Conv2d` zastąpiono **`CyConv2d`** (w tym `conv1`
i konwolucje w `BasicBlock`). Pozostałe elementy (BN, ReLU, shortcut, GAP,
`Linear`) pozostajone zostały bez zmian względem wersji bazowej.


## Wersje cykliczne: CyVGG-E i CyResNet-56

Wersje cykliczne powstają przez zastąpienie każdej `Conv2d` warstwą
`CyConv2d`. Interfejs (`kernel size`, `stride`, `padding`) jest zgodny
z `Conv2d`, więc topologia sieci i część klasyfikacyjna pozostają bez zmian.
`CyConv2d` opakowuje własną funkcję autograd (`CyConv2dFunction`) i
wywołuje rozszerzenie CUDA `CyConv2d_cuda.forward/backward(...)`. Wagi
mają kształt `[C_out, C_in, k, k]` i są inicjalizowane przez
`xavier_uniform_`. Moduł korzysta z dużego bufora roboczego na GPU
(opisanego w kodzie jako „Workspace for Cy-Winograd algorithm”).

W definicjach modeli nie występuje jawna oś „orientacja” ani osobny
pooling po orientacjach. Z punktu widzenia PyTorch parametry filtrów
zachowują standardowy kształt `[C_out, C_in, k, k]`. Mechanizmy
rotacyjne, o ile są użyte,  realizowane są w jądrze CUDA
`CyConv2d_cuda`, niewidocznym na poziomie kodu modeli.
W praktyce inwariancja po stronie modeli nie jest wprowadzana osobno:
`GAP` oraz ewentualne uśrednianie w klasyfikatorze działają tak samo jak
w wersjach bazowych i nie ma dodatkowego „poolingu po orientacjach”.

## Uzgodnienia I/O i selektor modeli

Aby uruchamianie eksperymentów było powtarzalne i przewidywalne, wszystkie modele korzystają
ze wspólnego stylu wejścia i wyjścia oraz prostego selektora architektury. 
Dzięki temu skrypty treningowe nie muszą znać szczegółów konkretnej sieci - 
wystarczy podać nazwę zbioru i skrót modelu, a resztą zajmuje się warstwa pomocnicza.

### Wejście i wyjście

Domyślnie używany jest jeden kanał wejściowy dla zbiorów w skali szarości 
(`mnist`, `mnist-custom`, `GTSRB-custom`, `LEGO`) oraz trzy kanały dla zbiorów RGB, 
takich jak `CIFAR-10/100` czy `GTSRB_RGB`.  
Liczba klas na wyjściu klasyfikatora ustalana jest automatycznie przez funkcję 
`get_num_classes(dataset)`. Przykładowo: `MNIST` i `CIFAR-10` mają po 10 klas, 
`GTSRB` - 43, `LEGO` - 50, a `CIFAR-100` - 100.

### Selektor modeli

Centralnym elementem jest fabryka modeli:

```python
model = get_model(model="cyresnet56",
                  dataset="GTSRB",
                  classify=True)
````

Argument `model` określa wariant architektury (`vgg*`, `cyvgg*`, `resnet*`, `cyresnet*`, np. `vgg19`, `cyvgg19`, `resnet56`, `cyresnet56`).
`dataset` wskazuje nazwę zbioru danych, na podstawie której ustalane są parametry `in_channels` i `num_classes`.
Flaga `classify=True` powoduje, że selektor domyka warstwę klasyfikacyjną (GAP + warstwa liniowa). 
Gdy ustawiona jest na `False`, zwracany jest sam „backbone”, co przydaje się przy analizie cech lub transferze uczenia.
Funkcja automatycznie dobiera liczbę kanałów wejściowych i klas, podstawia odpowiedni typ warstw (`Conv2d` lub `CyConv2d`)
i pozostawia resztę topologii bez zmian (BN/ReLU, bloki, GAP). Dzięki temu porównania modeli bazowych i cyklicznych są 
miarodajne - klasyfikator jest identyczny, a liczba parametrów i parametrów FLOPs zbliżona.

### Dodatkowe zasady bezpieczeństwa

Podczas wczytywania danych sprawdzana jest zgodność wymiarów tensora z oczekiwanym układem `C×H×W`. 
Jeśli liczba kanałów nie odpowiada zbiorowi, proces zostaje przerwany z komunikatem błędu, 
zamiast kontynuować z niepoprawnymi danymi.
Warstwy konwolucyjne (`Conv2d` i `CyConv2d`) są inicjalizowane metodą `xavier_uniform_`, 
natomiast warstwy liniowe - `kaiming_uniform_`, o ile konfiguracja nie definiuje inaczej. 
Taki sposób inicjalizacji zapewnia stabilny start niezależnie od wariantu modelu.
W modelach cyklicznych statystyki normalizacji batchowej (BN) liczone są wspólnie wzdłuż 
wymiaru „orientacja”. Dzięki temu żaden z kierunków nie jest faworyzowany, a zachowana 
zostaje ekwiwariancja rotacyjna.

### Przykład uruchomienia

Trening modelu **CyResNet56** na zbiorze **LEGO** (transformacja `linear-polar`, wariant `non_rotated`):

```bash
venv/bin/python main.py \
  --train \
  --model=cyresnet56 \
  --dataset=LEGO \
  --polar-transform=linearpolar \
  --data-dir=./data/LEGO/dataset_LEGO_non_rotated \
  --model-save-path=./saves/LEGO/LEGO-cyresnet56-linearpolar_dataset_LEGO_non_rotated.pt \
  --output-dir=./logs/json_LEGO/confusion_matrices/dataset_LEGO_non_rotated
```

Do testów tego samego modelu na zbiorze rotowanym wykorzystywana jest następująca komenda:

```bash
venv/bin/python main.py \
  --test \
  --model=cyresnet56 \
  --dataset=LEGO \
  --polar-transform=linearpolar \
  --data-dir=./data/LEGO/dataset_LEGO_non_rotated \
  --test-data-dir=./data/LEGO/dataset_LEGO_rotated_0_90 \
  --model-path=./saves/LEGO/LEGO-cyresnet56-linearpolar_dataset_LEGO_non_rotated.pt \
  --output-dir=./logs/json_LEGO/confusion_matrices/LEGO-cyresnet56-linearpolar_dataset_LEGO_non_rotated/dataset_LEGO_non_rotated_test_on_dataset_LEGO_rotated_0_90 \
  --use-prerotated-test-set
```

Dla wygody można użyć wersji ze zmiennymi środowiskowymi:

```bash
MODEL=cyvgg19
ACT=logpolar
TRAIN=dataset_LEGO_non_rotated
TEST=dataset_LEGO_range_0_180

VENVPY=venv/bin/python
DATA=./data/LEGO
SAVES=./saves/LEGO
LOGS=./logs/json_LEGO
CM=${LOGS}/confusion_matrices

${VENVPY} main.py --train \
  --model=${MODEL} \
  --dataset=LEGO \
  --polar-transform=${ACT} \
  --data-dir=${DATA}/${TRAIN} \
  --model-save-path=${SAVES}/LEGO-${MODEL}-${ACT}_${TRAIN}.pt \
  --output-dir=${CM}/${TRAIN}

${VENVPY} main.py --test \
  --model=${MODEL} \
  --dataset=LEGO \
  --polar-transform=${ACT} \
  --data-dir=${DATA}/${TRAIN} \
  --test-data-dir=${DATA}/${TEST} \
  --model-path=${SAVES}/LEGO-${MODEL}-${ACT}_${TRAIN}.pt \
  --output-dir=${CM}/LEGO-${MODEL}-${ACT}_${TRAIN}/${TRAIN}_test_on_${TEST} \
  --use-prerotated-test-set
```

### Nazwenictwo i dodatkowe opcje

Ścieżki i nazwy plików są spójne z konwencją launchera: 
`LEGO-<model>-<act>_<train>.pt`oraz katalogami wynikowymi w formacie `<train>_test_on_<test>`.
Parametr `--data-dir` zawsze wskazuje katalog treningowy (np. `non_rotated`), natomiast`--test-data-dir` - konkretny wariant testowy.
Opcja `--use-prerotated-test-set` powinna być włączona w sytuacji, gdy testy wykonywane są na już przygotowanych, rotowanych zestawach danych.
Jeśli potrzebny był zapis logu do pliku, dodane było standardowe przekierowanie wyjścia, np. `>> logs/plik.txt 2>&1`.

## Standardowe CNN

Jako bazy zastosowano **VGG** (wariant **E / VGG-19**) i **ResNet-56**. Konwolucje
`3×3` z `padding=1`, w VGG z okresowym `MaxPool2d(2)`, w ResNecie zaś jest zmniejszanie
rozdzielczości przez `stride=2`. Po części splotowej są **GAP** i **klasyfikator**
(w VGG są to dwie warstwy w pełni połączone `512→512→C` z dropoutem, aż w ResNecie jest to
`Linear 64→C`). Warianty VGG w kodzie występują także w wersjach `*_bn`
(z BatchNorm). Modele te są z natury **ekwiwariantne translacyjnie** (dobrze
znoszą przesunięcia), ale nie posiadają wbudowanego mechanizmu inwariancji
względem rotacji, stąd stanowią więc punkt odniesienia dla wersji cyklicznych
[@simonyan2014vgg; @he2016resnet].

## Rotacyjnie inwariantne sieci (CyResNet, CyVGG)

Wersje cykliczne (**CyVGG-E**, **CyResNet-56**) powstały przez **podmianę każdej warstwy
`Conv2d` na `CyConv2d`**. Architektura bloków, liczby kanałów, GAP i układ
klasyfikatora pozostane zostały bez zmian, co pozwala ma porównanie wpływu samej warstwy
splotowej. W kodzie modeli **nie ma jawnego wymiaru orientacji** ani
dedykowanego „poolingu po orientacjach”. Funkcjonalność związaną z rotacją
zrealizowana została w jądrze CUDA z użyciem `CyConv2d_cuda`, wywoływanym z poziomu `CyConv2d`
[@kim2020cycnn].

## Transformacje polarne: linearpolar vs logpolar

Przekształcenie do układu biegunowego mapuje obraz z współrzędnych (x, y)
na siatkę (ρ, φ), gdzie ρ opisuje odległość od środka, a φ kąt. Dzięki temu
obrót obrazu staje się przesunięciem wzdłuż osi φ, co upraszcza budowanie
reprezentacji ekwiwariantnych względem rotacji. W praktyce stosowane są dwie
siatki: linearpolar i logpolar.

W linearpolar przyrost po ρ i po φ jest równy w pikselach. Kąty są stabilne,
implementacja prosta, a odwzorowanie dobrze nadaje się do budowania
inwariancji rotacyjnej. Logpolar stosuje skalę logarytmiczną wzdłuż ρ, przez
co zmiany skali w obrazie zamieniają się w przesunięcia po osi promienia.
Takie odwzorowanie jest szczególnie użyteczne, gdy oprócz rotacji ważna jest
również zmienność skali [@reddy1996logpolar]. Wadą jest rosnąca gęstość
próbkowania w pobliżu środka, co zwiększa wrażliwość na wybór punktu
odniesienia i na szum.

Niezależnie od wariantu stosowana jest interpolacja biliniarna oraz
cykliczne dopełnianie po φ, aby nie powstawały artefakty na brzegach kąta.
Środek układu utrzymywany jest stały, w okolicy ρ = 0 warto wprowadzić
wygładzenie lub pominąć kilka najbliższych próbek, by ograniczyć wpływ
osobliwości i zniekształceń [@kim2020cycnn]. Wybór między linearpolar a
logpolar zależy więc od celu: gdy kluczowa jest odporność na obrót, wystarcza
siatka liniowa; gdy istotna jest także skala, korzystny bywa układ
logarytmiczny kosztem większej troski o okolice środka.

\newpage

# Implementacja i środowisko eksperymentalne

Wszystkie eksperymenty realizowane zostały w środowisku **PyTorch** z
rozszerzeniem CUDA dla warstwy cylindrycznej, dodatkowo zrobiona została automatyczna
optymalizacja hiperparametrów w **Optunie** dla wybranych modeli (w tym referencyjnego)
[@pytorch-docs; @akiba2019optuna]. Pipeline danych obejmuje warianty
IDX i NPY oraz generator zbiorów rotowanych i zbiorów połączonych ze sobą(merged).
Wyniki treningu oraz testów dla każdego modelu są zapisywana w formacie txt wraz z
macierzami pomyłek w fomratach png oraz npy. Następne dane są sprawdzane i automatyczne 
generowane są heatmapy train-test oraz ranking modeli.
Zapisy metryk i konfiguracji z optuny trafiają do plików **CSV** i **JSON**, co
ułatwia powtarzalność oraz porównywanie konfiguracji. Dodakowo najlepsze checkpointy 
dla danego przypadku są zapisane jako modele już przetrenowane .pt. 
Wykorzystywany był otymalizator **SGD** wraz  z *momentum* i *weight decay*. 
Zakresy i sposób doboru wartości hiperparametrów opisanostały w części poświęconej HPO.

## Warstwa `CyConv2d` (CUDA) oraz jej implementacja

Warstwa `CyConv2d` korzysta z rozszerzenia C++/CUDA kompilowanego jako `CyConv2d_cuda`.
Pliki źródłowe wykorzystywane do kompilacji to `cycnn.cpp` i `cycnn_cuda.cu`, ich budowanie
przy użyciu `setuptools` z `BuildExtension`. Dzięki temu wywołania tej biblioteki z poziomu Pythona
trafiają bezpośrednio do rdzeni CUDA.
W pliku `cycnn.cpp` udostępnione są funkcje `forward(...)` i `backward(...)`
z użyciem pybind11. Przyjmują one tensory `input`, `weight` oraz bufor
`workspace`, a także parametry geometrii: `stride`, `padding`, `dilation`.
Przed przekazaniem do implementacji CUDA sprawdzane jest, czy dane
znajdują się na GPU i mają ciągły układ w pamięci. Następnie wywoływane są
funkcje `cyconv2d_cuda_forward` i `cyconv2d_cuda_backward`.

Po stronie PyTorch warstwa jest opakowana w `CyConv2dFunction` z własnymi
`forward` i `backward`. Moduł `CyConv2d` przechowuje wagi o kształcie
`[C_out, C_in, k, k]` z inicjalizacją z użyciem `xavier_uniform_`. W metodzie `forward`
wywoływana jest `CyConv2dFunction.apply(...)`, do której przekazywane są
parametry kroku, dopełnienia i dylacji oraz wskaźnik do bufora roboczego.
Bufor `workspace` jest prealokowany na GPU jako tensor `float32` o rozmiarze
`1024*1024*1024` elementów, jest to w przybliżeniu około 4 GiB. W kodzie został opisany
jako miejsce pracy wariantu algorytmu Winograda. Rozwiązanie to pozwala skórcić czas obliczeń, ale
wymaga odpowiedniej ilości pamięci VRAM, przez co na kartach graficznych posiadających mniej VRAMU mogą
pojawić się błędy OOM.

Integracja z modelami zrobiona w sposób bezpośredni. We wszystkich miejscach, gdzie w
bazowych architekturach użyto `nn.Conv2d`, zarówno w `conv1`, jak i w
konwolucjach wewnątrz bloków, w wersjach **CyVGG** i **CyResNet** wstawiono
`CyConv2d`. Pozostałe elementy pozostają bez zmian: BatchNorm, ReLU, GAP i
klasyfikator działają tak samo jak w wersjach referencyjnych.
W samych definicjach modeli nie ma jawnej osi orientacji ani osobnego poolingu
po orientacjach. Z punktu widzenia PyTorch wagi mają klasyczny kształt
`[C_out, C_in, k, k]`. Jeśli pojawia się zachowanie związane z obrotami, to jest
ono realizowane w kodzie CUDA (`cycnn_cuda.cu`), do którego odwołują się
bindingi z `cycnn.cpp`.

## Python, PyTorch

Środowisko jest oparte na **Pythonie** i **PyTorchu** z akceleracją CUDA.
Modele definiowane są w stylu modułów `nn.Module`, zaś pętle uczące
korzystają z klasycznego układu: forward, obliczenie straty, backward,
aktualizacja optymalizatora. Zastosowane zostały usprawnienia backendowe
(`torch.backends.cudnn.benchmark = True`, czyli ustanienie F32 tam, gdzie jest to możliwe),
co pozwala skrócić czas uczenia na nowszych GPU. W modelach jest używany optymalizator
**SGD** z *momentum* i *weight decay*, a harmonogram uczenia opiera się
na `ReduceLROnPlateau`. Kod modeli jest kompatybilny z ekosystemem i API
PyTorcha, więc warstwy bazowe można bez problemowo wymieniać na odpowiedniki
cylindryczne bez zmian w pozostałych fragmentach sieci neuronowej.

## Struktura projektu

Projekt podzielony jest na dwie wzajemnie uzupełniające się części. Repozytorium
treningowe zawiera modele (**VGG, ResNet** oraz **CyVGG, CyResNet**),
warstwę `CyConv2d` z rozszerzeniem **C++/CUDA**, loader danych dla
formatów **IDX** i **NPY**, transformacje i mapowania polarne oraz
główny skrypt `main.py` oraz launchery dla poszczególnych datasetów. 
Znajdują się tam także pliki uruchamiające Optunę i konfiguracje eksperymentów.

Repozytorium zarządzające skupia narzędzia do przygotowania danych,
rotacji i łączenia zbiorów, a także do analizy wyników. Wykorzystywany jest
interfejs CLI (Typer), który buduje zbiory, wczytuje logi i artefakty,
zapisuje metryki do **SQLite** oraz generuje mapy ciepła train-test i
zestawienia rankingowe. Artefakty są porządkowane w powtarzalnej
strukturze katalogów: `logs/` dla przebiegów, `saves/` dla wag `.pt`,
foldery z macierzami pomyłek w wariancie `.npy` i `.png`, a wyniki
zbiorcze w `results/`.

## Automatyzacja: skrypty trenowania, testowania, ewaluacji

Trenowanie modeli uruchamiane z wykorzystaniem skryptu launcher_<dataset>.py, który 
wywołuje `main.py` z odpowiednimi parametrami dla danego
modelu, zbioru, wariantu przekształceń i ustawień treningu. W trakcie
ewaluacji zapisywana jest macierz pomyłek oraz podstawowe metryki
dokładności. Repozytorium orkiestrujące dostarcza komendy CLI do pełnego
przebiegu. Najpierw `preprocess` przygotowuje zbiory rotowane i zestawy
połączone. Następnie uruchamiany jest trening i test, po czym `ingest`
zbiera logi oraz wyniki i zapisuje je do bazy **SQLite**. Komenda
`check-logs` weryfikuje kompletność przebiegów. Moduły `analyze` i
`matrix-analyzer` tworzą mapy ciepła train-test, agregują macierze
pomyłek, liczą statystyki i budują rankingi modeli. W wybranych
konfiguracjach dołączona jest **Optuna**, która automatycznie stroi
hiperparametry i zapisuje najlepsze konfiguracje wraz z wynikami do
plików **CSV/JSON**.

\newpage

## Automatyczna optymalizacja hiperparametrów z wykorzystaniem Optuny

W celu poprawy jakości trenowanych modeli zastosowana została automatyczna
optymalizacja hiperparametrów. Ręczne dobieranie wartości takich jak *learning
rate*, *momentum* czy *weight decay* jest czasochłonne, podatne na błędy i bardzo
często prowadzi do ustawień dalekich od optimum. Optymalne parametry zależą od
architektury sieci, charakteru zbioru danych oraz użytych przekształceń
polarnych. Z tego powodu wykorzystana została biblioteka **Optuna**
[@akiba2019optuna], nowoczesne narzędzie do strojenia hiperparametrów
(*Hyperparameter Optimization, HPO*).

W eksperymentach użyty został algorytm próbkowania **TPE (Tree structured
Parzen Estimator)** oraz mechanizm **pruning Median**, który umożliwia
wcześniejsze przerywanie prób o niskiej jakości. Dzięki temu czas potrzebnych obliczeń
został skrócony o około  trzydzieści procent. Wyniki każdej próby są zapisywane
do plików **CSV** i **JSON**, co ułatwia późniejszą analizę oraz wierne
odtworzenie najlepszych konfiguracji.

Proces optymalizacji uruchomiony został dla czterech modeli
(**ResNet56**, **VGG19**, **CyResNet56**, **CyVGG19**) oraz dla dwóch wariantów
transformacji (*linear polar*, *log polar*). Optymalizacja prowadzona była na
zestawach oznaczonych jako *non_rotated*. W każdej konfiguracji wykonano
25 prób z limitem 10 epok na próbę, aby ograniczyć czas trwania
eksperymentów. Przeszukiwane były następujące zakresy:  \
*learning rate* od `1e-4` do `5e-2` w skali logarytmicznej,  \
*weight decay* od `1e-7` do `1e-3` w skali logarytmicznej,  \
*momentum* od `0.85` do `0.99` w skali liniowej.  \

Zastosowanie Optuny pozwoliło osiągnąć wyższą dokładność niż w ustawieniach
dobieranych ręcznie. Najczęściej wybierane wartości *learning rate* mieściły
się w przedziale od `0.001` do `0.01`, co jest spójne z obserwacjami występującymi w
literaturze. Korzyści były widoczne także w prostszych zbiorach, takich jak
MNIST oraz GTSRB, a w bardziej złożonych scenariuszach optymalizacja ograniczyła
liczbę arbitralnych decyzji i obniżyła koszt obliczeń.

Włączenie automatycznej optymalizacji do cyklu eksperymentalnego zwiększyło
wiarygodność wyników. Otrzymane wyniki dla modeli oparte są na systematycznym dostrajaniu
zgodnym z aktualnym stanem wiedzy, a nie ślepych na pojedynczych ręcznych próbach.

## Obsługa GPU, Docker, WSL
Środowisko uruchomieniowe zostało zorganizowane tak, aby połączyć
**wydajność GPU**, **powtarzalność kontenerów** oraz **spójność pracy w 
przypadku pracy na Windows przez WSL2**. 
Treningi i testy uruchamiane są w obrazie Dockera
z przypiętymi wersjami **CUDA** i **cuDNN**, a ten sam obraz jest
wykorzystywany w **WSL2**, co eliminuje różnice między stacjami roboczymi.
Dane i artefakty (logi, checkpointy, macierze pomyłek) są montowane jako
woluminy, więc katalogi z wynikami pozostają niezmienne między sesjami i
hostami. Dobór urządzeń kontrolowany jest przez `CUDA_VISIBLE_DEVICES`,
a ścieżki wejścia i wyjścia mają jednolitą konwencję, co upraszcza
późniejszą analizę i replikację. Monitorowanie obciążenia oraz VRAM
odbywa się narzędziem `nvidia-smi`, natomiast szczegóły konfiguracji
przepływu danych i precyzji obliczeń opisane zostały poniżej.

### Obsługa GPU
Środowisko uruchomieniowe wykorzystuje akcelerację **CUDA** na kartach
**RTX 3070 Ti 8 GB** oraz **RTX 3060 12 GB**. Włączone są optymalizacje
backendowe: `torch.backends.cudnn.benchmark = True` oraz tryb **TF32**
na architekturze Ampere. Transfery między CPU i GPU realizowane są z
`pin_memory=True` w `DataLoader` oraz `non_blocking=True` przy
kopiowaniu tensora na urządzenie, co zmniejsza narzut I/O. W miejscach,
gdzie to bezpieczne do użycia, wykorzystywana jest **mieszana precyzja** z
`torch.cuda.amp` i `GradScaler`. Monitorowanie obciążenia i pamięci
odbywa się przez `nvidia-smi`. Należy uwzględnić wymóg pamięci VRAM dla
warstwy `CyConv2d` (bufor roboczy ~4 GiB na GPU), co sprawia, że 
minimalna wymagana ilośc VRAMu wynosi ~6GB.

### Docker
**Docker** zapewnia powtarzalne środowisko z obsługą GPU. Obraz zawiera
**PyTorch**, **CUDA**, **cuDNN** oraz zależności projektu. Uruchomienie
odbywa się z wykorzystaniem **nvidia-container-toolkit**. Woluminy mapują katalogi z
danymi i wynikami:

```bash
docker run \
  --gpus all \
  -it --rm \
  -v /path/to/data:/workspace/data \
  -v /path/to/results:/workspace/results \
  cycnn:latest

```
### WSL2
**WSL2** umożliwia pracę z subsystemem Linux w środowisku **Windows** 
z dostępem do tego samego obrazu **Dockera** oraz tej samej konfiguracji. 
Wyniki i pliki pomocnicze (artefakty) mogą być kopiowane poprzez ścieżkę `\wsl$` 
do lokalnych katalogów w celem dalszej analizy. 
Zachowana zostaje spójność nazw katalogów i plików, co ułatwia późniejszy 
**import danych** do bazy **SQLite** oraz generowanie raportów.  
W przypadku różnic w separatorach wykorzystywanych do tworzenia ścieżek, logika 
zastosowanych narzędzi unika operacji zależnych od systemu operacyjnego, 
a ścieżki są przekazywane jawnie w interfejsie wiersza poleceń (**CLI**).


## Organizacja logów, modeli, confusion matrixów

Artefakty przeprowadzonych eksperymentów są porządkowane w określonej stałej strukturze katalogów.
Dzięki temu skrypty analityczne mogą automatycznie odnajdywać logi,
modele i macierze pomyłek, zestawiać wyniki i budować heatmapy
train-test. Ułatwia to też nawigacje po wynikowych danych.

```
CyCNN-Enhanced-develop/
|-- ReadMe.md
|-- LICENSE
|-- .gitignore
|-- cycnn-extension/                  # rozszerzenie C++/CUDA dla CyConv2d
|   |-- cycnn.cpp
|   |-- cycnn_cuda.cu
|   `-- setup.py
`-- cycnn/                            # trenowanie i testowanie modeli
    |-- main.py
    |-- utils.py
    |-- data.py
    |-- custom_loader.py
    |-- image_transforms.py
    |-- requirements.txt
    |-- launcher_MNIST.py
    |-- launcher_GTSRB.py
    |-- launcher_GTSRB_RGB.py
    |-- launcher_LEGO.py
    |-- optuna_launcher.py
    |-- optuna_driver_universal.py
    |-- optuna_checker.py
    |-- optuna_mnist.py
    |-- train_test_scenarios_MNIST.json
    |-- train_test_scenarios_GTSRB.json
    |-- train_test_scenarios_GTSRB_RGB.json
    |-- train_test_scenarios_LEGO.json
    |-- models/
    |   |-- getmodel.py
    |   |-- cyconvlayer.py
    |   |-- cyresnet.py
    |   |-- cyvgg.py
    |   |-- resnet.py
    |   `-- vgg.py
    |-- logs/                         # logi i (domyślnie) macierze pomyłek
    |   `-- .gitkeep
    `-- saves/                        # checkpointy (.pt)
        |-- .gitkeep
        `-- MNIST/                    # przykładowy podkatalog (opcjonalny)


```
### Logi
**Logi (`logs/`)** zapisywane są podczas uruchomienia z flagą `--redirect`.
Plik trafia na dysk pod wzorcem:
```
cycnn/logs/<fname>.txt
```
gdzie nazwa `<fname>` budowana jest z następujących składników:
```
<dataset>-<model>[-<polar_transform>][-<augmentation>][-rotation_from_scenarios]
```
Przykład używany w eksperymentach:
\path{cycnn/logs/mnist-custom-cyresnet56-linearpolar_merged_datasets_merged_range_0_180_plus_non_rotated_train.txt}
### Modele
**Modele (`saves/`)** zawierra on najlepsze checkpointy w formacie `.pt`.
Zapis następuje przy poprawie wyniku lub zgodnie z polityką zapisu w skrypcie.
Domyślna ścieżka ma postać:
```
cycnn/saves/<fname>.pt
```
Jeśli podano `--model-save-path`, checkpoint zostaje zapisany dokładnie w tej
ścieżce, z pominięciem wzorca domyślnego.

### Macierze pomyłek

Podczas testowania modelu zapisywane są **macierze pomyłek** w kilku wariantach oraz zestaw podstawowych metryk.  
Pliki generowane są zarówno w formacie graficznym (`.png`), jak i numerycznym (`.npy` / `.csv`).

Domyślnie, jeśli nie podano parametru `--output-dir`, wyniki trafiają do katalogu zależnego od pary train–test:
```
cycnn/logs/<train_set>*test_on*<test_set>/
```
W przypadku jawnie wskazanego katalogu (`--output-dir <DIR>`), wszystkie pliki zapisywane są w jego obrębie:
```
<DIR>/
```
#### Zawartość katalogu wynikowego
Przykładowa struktura po zakończeniu testu:
```
confusion_matrix.npy
confusion_matrix.png
confusion_matrix_row_norm.npy
confusion_matrix_row_norm.png
confusion_matrix_counts.png
metrics.csv
```
Macierze o nazwie `confusion_matrix.*` zawierają surowe wyniki klasyfikacji.
Wersje z dopiskiem `_row_norm` przedstawiają wartości znormalizowane względem wierszy 
(czyli dla każdej klasy suma = 1), co ułatwia porównanie proporcji błędów.
Plik `confusion_matrix_counts.png` wizualizuje liczby bez normalizacji, plik `metrics.csv` 
zawiera zbiorczą dokładność (overall_accuracy) i dokładności per klasa (unormowaną w przedziale od 0 do 1).
Plik wyląda w ten sposób:
```
overall_accuracy,0.947268

class, recall
0,     1.000000
1,     0.983333
...
42,    0.988889
```
Takie uporządkowanie pozwala łatwo analizować wyniki eksperymentów, niezależnie od liczby wariantów zbiorów i modeli.

###Scenariusze train-test (JSON)
**Scenariusze train-test (JSON)** zawierają gotowe listy par zbiorów, których
nazwy odpowiadają rzeczywistym ścieżkom na dysku. W repozytorium znajdują się następujące scenariusze:
```
train_test_scenarios_MNIST.json
train_test_scenarios_GTSRB.json
train_test_scenarios_GTSRB_RGB.json
train_test_scenarios_LEGO.json
```

\newpage

# Eksperymenty

Część eksperymentalna została zbudowana tak, aby wprost porównać wersje
bazowe (VGG-19 E, ResNet-56) z ich odpowiednikami rotacyjnymi (CyVGG-19 E,
CyResNet-56) przy niezmienionym budżecie parametrów i podobnym FLOPs. Głównym
celem jest sprawdzenie, jak wprowadzenie osi „kąt” i cyklicznego
dopełniania po $\varphi$ wpływa na jakość i stabilność względem obrotów,
a także czy zyski utrzymują się przy zmianie rozkładu kątów między
treningiem a testem.

Zakres obejmuje cztery zbiory: **MNIST**, **GTSRB Gray**, **GTSRB RGB**
i **LEGO**. Dla wszystkich zastosowano spójny preprocessing, ustaloną
rozdzielczość wejścia oraz normalizację. Dane przygotowano w dwóch
formatach wejściowych (IDX i NPY), a następnie wygenerowano warianty
obrotowe w dwóch trybach: **kąty stałe** oraz **przedziały kątowe**.
Na tej podstawie zbudowano również presety złączone (np.
`merged_fixed_30`, `merged_range_full_0_360` oraz wersje z dopiskiem
`+ non_rotated`) tak, aby systematycznie zbadać uogólnianie „trenuj na
X, testuj na Y”.

Aby wyeliminować wpływ przypadkowego doboru hiperparametrów, kolejnym kroku
kroku wykonano **automatyczną optymalizację** (Optuna, TPE + pruning) na
przypadkach *non_rotated*, przy czym też został przeprowadzony na zbiorze GTSRB trening
tak aby sprawdzić czy da się uzyskać większą odporność na obrót
samą zmianą sposobu wybierania nalepszego checkpointu. Trening
pozostaje *non_rotated* (bez obrotów w danych), ale walidacja i kryterium
wyboru modelu patrzą już na **zachowanie względem kątów**.
Wyniki pokazały, że początkowe parametry zostały ustawione właściwie, 
przez co są one wykorzystywane w kolejnych eksperymentach.
Protokół trenowania jest stały w całej serii (liczba epok, rozmiar batcha, scheduler, 
optymalizator`SGD` z `momentum` i `weight_decay`), a różnice dotyczą
wyłącznie architektury splotu (`Conv2d` → `CyConv2d`) i przygotowania 
danych poprzez dodanie rotacji.

Uruchomienia są orkiestrane na podstawie plików **JSON** opisujących scenariusze:
każdy zestaw treningowy ma przypisaną listę zestawów testowych
(ścieżki 1:1 z drzewem katalogów). Rozwiązanie to upraszcza replikację,
pozwala pozwala łatwo utworzyć macierze porównań oraz utworzyć
wizualizacje w postaci map ciepła „train-test”.

Obliczenia realizowane są na GPU **NVIDIA RTX 3070 Ti** i **RTX 3060**.
Akceleracja opiera się na wykorzystaniu **CUDA**, **cuDNN** i **cuBLAS**, 
dodatkowo włączony jest tryb **TF32** (Ampere), a tam gdzie to bezpieczne 
używana jest mieszana precyzja przez `torch.cuda.amp`. 
Należy uwzględnić bufor roboczy`CyConv2d` (~4 GiB VRAM). 
Ziarna generatorów pseudolosowych są z góry ustawiane dla Pythona, NumPy i PyTorcha/ 
Ustawiene został cudnn.benchamar na wartość true (`cudnn.benchmark = True`)
celem redukcji czasu trenowania, co nie zaburza porównywalności wyników.

Wyniki zapisywane są w spójnej strukturze zawierającej: logi z przebiegów, najlepsze
checkpointy `.pt`, macierze pomyłek w formatach `.npy` i `.png`,
zestawienia CSV oraz wpisy w bazie **SQLite**. Na tej podstawie
budowane są rankingi oraz statystyki zbiorcze (średnia, mediana,
odchylenie standardowe). Poniżej opisany został sposób definiowania scenariuszy,
procedurę pomiaru skuteczności oraz metodę agregacji metryk.

\newpage

## Scenariusze trenowania/testowania (opis JSON)

Scenariusze zdefiniowane są przez pliki JSON, które mapują **zestawy
treningowe** na listy**zestawów testowych**. Klucze i wartości są
po prostu ścieżkami katalogów w strukturze danych. Dzięki temu w łatwy sposób
powiązane zostały nazwy w  pliku JSON z realnymi ścieżkami na dysku 
i możliwe było przeprowadzenie serii eksperymentów bez ręcznej konfiguracji czy uruchamiania.

Przykładowy fragment pliku JSON ze scenariuszami train-test dla zbioru LEGO:

```json
{
  "dataset_LEGO_non_rotated": [
    "dataset_LEGO_non_rotated",
    "merged_datasets/merged_fixed_30",
    "merged_datasets/merged_fixed_45_plus_non_rotated",
    "rotated-90",
    "rotated-90-120"
  ],
  "merged_datasets/merged_range_full_0_360_plus_non_rotated": [
    "dataset_LEGO_non_rotated",
    "merged_datasets/merged_range_0_180",
    "merged_datasets/merged_range_180_360",
    "rotated-120",
    "rotated-210-240"
  ]
}
```

Listy testowe budowane zostały według określonych reguł ich generatorowania. 
Zawsze zawierają one co najmniej zestaw bazowy bez rotacji oraz sam zestaw treningowy.
Pozostałe pozycje dobierane są z puli wariantów obrotowych i presetów
łączonych do ustalonego limitu. Wygenerowane nazwy są zgodne
z konwencjami katalogów, które powstają podczas preprocessingu.

Podczas procedury treningu-testowania pętla uruchomieniowa 
odczytuje scenariusz, a następnie wykonuje następujaca sekwencję:
trening na danym **train_set**, a następnie testy na wszystkich
**test_set** z listy. Poniżej znajduje się uproszczony szkic logiki, którą realizują skrypty:

```text
for train_set in scenarios:
  model = build_model(...)
  fit(model, train_set, ...)
  for test_set in scenarios[train_set]:
    acc, cm = evaluate(model, test_set)
    save_metrics_and_artifacts(acc, cm, train_set, test_set, model)
```

Takie podejście porządkuje eksperymenty. Pozwala też tworzyć mapy
„trenuj na X, testuj na Y” oraz automatycznie budować rankingi modeli
wspólne dla całego zbioru scenariuszy.

## Pomiar skuteczności (accuracy, macierze pomyłek)

Skuteczność raportowana jest jako **dokładność top-1** dla każdej pary
train-test. Wartość wyznaczana jest z **macierzy pomyłek** o rozmiarze
$C \times C$, zapisywanej jako `confusion_matrix.npy`.

**Micro-accuracy** (zastosowana w pracy):
$$
\mathrm{Acc}_{\mathrm{micro}}
=\frac{\sum_{k=1}^{C}\mathrm{TP}_k}
       {\sum_{k=1}^{C}\big(\mathrm{TP}_k+\mathrm{FP}_k+\mathrm{FN}_k\big)}
=\frac{\operatorname{tr}(CM)}{\sum CM}.
$$

**Macro-accuracy** (opcjonalnia nie zastosowana w pracy) - średnia z dokładności klas:
$$
\mathrm{Acc}_{\mathrm{macro}}
=\frac{1}{C}\sum_{k=1}^{C}
  \frac{\mathrm{TP}_k}{\mathrm{TP}_k+\mathrm{FN}_k}.
$$

Macierz pomyłek zapisywana jest w dwóch postaciach: jako surowe wartości
(`.npy`) do analizy oraz wizualizacja (`.png`) do raportów (z opcją
normalizacji wierszowej albo globalnej). Heatmapa **train-test** (PNG)
pokazuje jakość dla układu „trenuj na X, testuj na Y”. Dokładności
per-klasa są wykorzystywane w wykresach porównawczych.


## Śledzenie metryk: średnia, mediana, odchylenie standardowe

Dla każdego modelu agregowane są wyniki z przypisanych scenariuszy
kątowych, raportowane są: **mean**, **median**, **min**, **max**,
**std**. Dodatkowo liczone są wskaźniki stabilności: **robust mean**
(średnia ucięta 10%) oraz **IQR**. Wyznaczany jest także
**gap train-test** - różnica między przypadkami „train-like”
(np. zawierającymi `non_rotated` albo `plus_non_rotated`), a resztą.

Wyniki i metadane trafiają do **SQLite** (przykładowo do tabel `evaluations`,
`training_runs`) oraz do **CSV** zapisywanego pod następującą ścieżką 
`results/exports/<DATASET>/<micro|macro>/...`, co ułatwia późniejsze
filtrowanie po: modelu, transformacji i zbiorze, a także łatwiejszą 
budowę rankingów i wykresów.

## Analiza skuteczności względem rotacji

Nazwy scenariuszy (`rotated-a[-b]`, `range_a_b`, `full_0_360`,
`non_rotated`) determinują przedziały kątów, na których podstawie wyznaczane są środki
przedziałów oraz różnica kątowa $\Delta\theta$ na okręgu z wrap-around
(zakres $[0^\circ, 180^\circ]$). 
Następnie obliczane są następujące parametry:  \

- krzywe $Acc(\Delta\theta)$ z koszykowaniem co
  $\theta_{\text{step}}=15^\circ$,  \
- $AUC_{\theta}$ (pole pod krzywą, obliczane metododa trapezową wraz normalizacja przez
  $180^\circ$),  \
- $Acc_{\min}$ (najgorszy koszyk),  \
- $SD_{\theta}$ (odchylenie między koszykami).  \

Eksport tych parametrów odbywa się do `delta_curves/acc_vs_delta_<MODEL>.csv` oraz
`auc_theta_ranking.csv`. Warto podkreślić, że spójny krok kątowy i jednolite zasady wrap-around
zapewniają porównywalność między modelami.


## Ranking modeli

Rankingi modeli są tworzone na podstawie następujących dwóch sposobów:

**Quality-only.** Sortowanie po `avg`, a przy remisach kolejno:
`std` (niższe lepsze), `min`, `median`, `max`, `robust_mean`, `IQR`
(niższy lepszy). Ich eksport odbywa się do pliku `ranking_quality.csv`.

**Time-aware.** Uwzględnianie kosztu czasowego:  \
- `avg/time` oraz `min/time` (`ranking_timeaware_avgperf.csv`),  \
- wariant zbalansowany `avg` vs `avg_perf`
  (`ranking_timeaware_balanced.csv`),  \
- wariant zbalansowany tylko per-time
  (`ranking_balanced_per_time.csv`).  \

Parametry i FLOPs nie są obecnie uwzględniane. Dzięki spójnym ścieżkom
artefaktów i zapisowi metryk do CSV/SQLite porównywanie wariantów
bazowych i rotacyjnych pozostaje powtarzalne i przejrzyste.

# Interpretacja wizualizacji wyników

**Macierz pomyłek.** Jasna, ciągła przekątna oznacza wysoką poprawność.
Pojawiające się skupiska większe niż kilka procent poza przekątną wskazują 
systematyczne błędy między konkretnymi klasami. W zadaniach wrażliwych na 
orientację błąd ma często charakter „lustrzany” dla par klas o podobnych kształtach.

**Mapa train-test.** Wiersze odpowiadają rozkładom kątów użytym w treningu,
kolumny rozkładom w teście. Każda komórka to top-1 accuracy. Jednolite
jasne pasma oznaczają stabilne uogólnianie poza rozkład treningowy.
Ciemniejsze obszary przy skrajnych kolumnach sugerują spadek jakości dla
dużych różnic kątowych.

**Wyniki per klasa vs kąt.** Wykresy z wynikami w koszykach kątowych
pozwalają wskazać klasy wyraźnie tracące jakość wraz ze zmianą orientacji
oraz klasy odporne na obrót. To ułatwia decyzje, czy potrzebna jest
dodatkowa augmentacja, czy zmiana części „rotacyjnej” architektury.


## Zakres odpowiedzialności narzędzi

`matrix_analyzer.py` oblicza micro i macro accuracy, agreguje statystyki
i czasy, wyznacza metryki zależne od kąta, eksportuje pliki CSV i drukuje
rankingi. Obrazy PNG (macierze, mapy train-test, wykresy per klasa vs kąt)
są generowane w modułach ewaluacyjnych testów. Analizator korzysta z tych
samych danych w formacie NPY i uzupełnia je o tabele potrzebne do raportu.


## Struktura wyników i artefaktów

Wyniki i atrefakty były tworzone i przechowywane zgodnie z poniższa strukturą.

```
Results/
db/experiment_logs.db
logs/matrix_<DATASET>*<METRYKA>*<ZNACZNIK_CZASU>.log
results/exports/<DATASET>/<micro|macro>/
ranking_quality.csv
ranking_per_time.csv
ranking_timeaware_avgperf.csv
ranking_timeaware_balanced.csv
ranking_balanced_per_time.csv
train_test_gap.csv
delta_curves/
acc_vs_delta_<MODEL>.csv
auc_theta_ranking.csv

```


## Ograniczenia i kierunki rozwoju

Analizator nie tworzy obrazów PNG, on koncentruje się na metrykach i wynikowych plikach
CSV (m.in. metryki kątowe, wskaźniki stabilności, rankingi). Obrazy
powstają w innych częściach pipeline’u i są używane w interpretacji.
W kolejnych iteracjach warte rozważenia są:  \
- integrację generowania obrazów w tym samym narzędziu,  \
- dołączenie parametrów i FLOPs do rankingów  \
- zapis dokładności per klasa w CSV równolegle z wizualizacjami  \


\newpage

# Porównanie wyników

Poniższy rozdział zbiera pełny obraz jakości i stabilności badanych
konfiguracji. Uwzględnione są cztery zbiory (MNIST, GTSRB, GTSRB_RGB,
LEGO), dwie rodziny modeli bazowych (VGG-19, ResNet-56) oraz ich
odpowiedniki cykliczne (CyVGG-19, CyResNet-56). Porównywane są dwa
warianty odwzorowania (linear-polar, log-polar). Prezentowane metryki obejmują
dokładność top-1 (micro), wskaźniki odporności na kąt (krzywe
Acc(Δθ), AUC_θ, worst-case), a także ujęcie per-time
(jakość na jednostkę czasu trenowania). Wnioski najpierw syntetycznie
opisują zachowanie całych rodzin, następnie omawiają specyfikę poszczególnych
zbiorów i kończą się prostą regułą wyboru konfiguracji pod ograniczenia
praktyczne.

## Zakres i procedura

Eksperymenty przeprowadzono według jednolitego protokołu trenowania,
z zachowaniem tych samych zasad walidacji i budżetu obliczeń.
Dla każdego modelu wykonano serię scenariuszy train–test obejmujących
różne warianty rotacyjne: zbiory bez rotacji, zbiory o stałych kątach
oraz zbiory o kątach losowanych z przedziałów.  
Wynik testu zapisywano w postaci macierzy pomyłek `C×C`, a dokładność
micro liczono bezpośrednio z tej macierzy.

Stabilność względem rotacji oceniana była za pomocą następujących wykresów:
- **heatmap train–test**, pokazujących wzajemną generalizację pomiędzy
zakresami kątów treningu i testu,
- **krzywych Acc(Δθ)**, które agregują dokładność względem różnicy
rozkładów kątów pomiędzy treningiem i testem.  

Metryka **AUC_θ** odpowiada znormalizowanemu polu pod krzywą Acc(Δθ),
obliczanemu metodą trapezów po ujednoliconych przedziałach Δθ i
skalowanemu do zakresu 0–1.  
Wariant *per-time* uzyskiwano przez odniesienie średniej dokładności
do całkowitego czasu trenowania.

\newpage

## Metryki i interpretacja
Micro-accuracy oddaje skuteczność „po wszystkich próbkach”, więc jest
odporna na nierówne liczebności klas w stopniu typowym dla zadań
klasyfikacji. AUC_θ syntetyzuje zachowanie krzywej Acc(Δθ):
im wyższe, tym reprezentacja bardziej wyrównana względem kątów. Wartość
worst-case wyłapuje najtrudniejszy punkt krzywej, który jest ważny, gdy
liczy się niezawodność w całym zakresie. Metryka per-time pomaga wybrać
wariant pod ograniczenia budżetu czasu i energii. Wykresy heatmap
train-test uzupełniają metryki numeryczne i pozwalają wzrokowo ocenić,
czy model „grzeje” także w kolumnach odległych kątowo od treningu.

*[Rys. 2: Krzywe Acc(Δθ)  dla zbioru GTSRB RGB dla modeli ResNet56(logpolar) oraz CyResNet56(logpolar)]*  \
![Rys. 2: Ranking AUCθ (barplot) dla zbioru GTSRB RGB](media%2Fassets%2Fplots%2Facc_vs_delta_GTSRB_RGB_ResNet_log_vs_CyResNet_log.png)  \

Rys. 2: Krzywe Acc(Δθ) na zbiorze GTSRB_RGB: przebieg cykliczny jest wyższy i bardziej płaski w całym zakresie [0°, 180°].  \
Wariant cykliczny (CyResNet56-log) utrzymuje stabilną dokładność ~0.9 w całym zakresie [0°, 180°], 
podczas gdy wariant bazowy (ResNet56-log) traci ok. 0.02 na krawędziach, co przekłada się na różnicę AUC_θ ≈ +0.03 na korzyść wariantu cyklicznego.

## Obraz ogólny: rodziny i transformacje

Modele cykliczne konsekwentnie podnoszą średnią jakość i stabilność
rotacyjną względem klasycznych odpowiedników. Najsilniejszy efekt widać
na GTSRB_RGB, gdzie różnica w średniej jakości sięga dziesiątek punktów
procentowych. Na GTSRB i LEGO zyski są wyraźne, na MNIST umiarkowane
(blisko sufitu), ale krzywe Acc(delta_theta) i AUC_theta nadal przemawiają
za rozwiązaniami cyklicznymi. Wariant log-polar zwiększa odporność na
duże różnice kątów, co podnosi AUC_theta i worst-case. Wariant
linear-polar częściej wygrywa w ujęciu per-time, zwłaszcza w rodzinie
ResNet. Wspólny wzorzec to lepsza generalizacja poza rozkład treningowy:
modele cykliczne uczone non_rotated utrzymują wysokie wartości także dla
kolumn testowych z odległymi kątami.

*[Rys. 3: Ranking AUCθ (barplot) dla zbioru LEGO]*  \
![Rys. 3: Ranking AUC_\theta (barplot) dla zbioru LEGO](media%2Fassets%2Fplots%2Fauc_theta_ranking_LEGO_micro.png)
Rys. 3. Globalna stabilność rotacyjna (AUCθ) dla datasetu LEGO: najwyżej plasują się konfiguracje cykliczne, zwłaszcza z log-polar.

## VGG vs CyVGG

Wersje CyVGG zyskują przede wszystkim na log-polar: rośnie AUC_theta,
spłaszczają się krzywe Acc(delta_theta), a worst-case przestaje być
„wąskim gardłem”. Na GTSRB_RGB różnica względem VGG-19 jest największa,
co potwierdzają zarówno liczby, jak i heatmapy. Na LEGO CyVGG-log
bywa liderem stabilności, a per-time utrzymuje rozsądny poziom. Na
MNIST różnice są mniejsze, ale nadal widoczne w przebiegach względem
delta_theta.
\newpage

*[Rys. 4 - Acc(Δθ) - VGG19-log vs CyVGG19-log (GTSRB_RGB)]*  \
![Rys. 4 - „Acc(Δθ) - VGG19-log vs CyVGG19-log (GTSRB_RGB)”](media%2Fassets%2Fplots%2Facc_delta_gtsrb_rgb_vgglog_vs_cyvgglog.png)  \
Rys. 4 - „CyVGG-log wyraźnie przewyższa VGG-log dla całego zakresu Δθ.”  \
*[Rys. 5 - Macierze pomyłek VGG19-log vs CyVGG19-log (GTSRB_RGB) trenowane na rotated-90-120)]*  \
![Rys. 5 - „Macierze pomyłek VGG19-log vs CyVGG19-log (GTSRB_RGB) trenowane na rotated-90-120)”](media%2Fassets%2Fplots%2Fconfmat_GTSRVB_RGB_VGG19_vs_CyVGG19_rot90-120.png)  \
Rys. 5 - „Wersja cykliczna ma bardziej skupioną przekątną i mniej symetrycznych rozlań błędów.”
\newpage

## ResNet vs CyResNet

W rodzinie ResNet linear-polar daje świetny kompromis jakości i per-time,
sprawdzający się na LEGO i MNIST. Wariant log-polar podnosi stabilność
dla dużych delta_theta i często wygrywa AUC_theta na GTSRB oraz na RGB.
W praktyce CyResNet bywa „bezpiecznym domyślnym wyborem”: średnia jakość
jest wysoka, krzywe są równe, a koszt obliczeń pozostaje akceptowalny.

*[Rys. 6 - „Acc(Δθ) - ResNet56-linear vs CyResNet56-linear (LEGO)”]*  \
![Rys. 6 - „Acc(Δθ) - ResNet56-linear vs CyResNet56-linear (LEGO)”](media%2Fassets%2Fplots%2Facc_vs_delta_LEGO_ResNet_linear_vs_CyResNet_linear.png)  \
Rys. 6: „CyResNet56-linear łączy wysoką średnią z dobrą efektywnością per-time.”  \
\newpage
*[Rys. 7 - „Acc(Δθ) - ResNet56-log vs CyResNet56-log (GTSRB_RGB)”]*  \
![Rys. 7 - „Acc(Δθ) - ResNet56-log vs CyResNet56-log (GTSRB_RGB)”](media%2Fassets%2Fplots%2Facc_vs_delta_LEGO_ResNet_linear_vs_CyResNet_linear.png)  \
Rys. 7: „Wariant log-polar u CyResNet zapewnia najwyższe AUCθ na RGB.”  \

## CyVGG vs CyResNet

CyResNet częściej prowadzi w średniej i per-time, zwłaszcza z
linear-polar. CyVGG-log bywa liderem stabilności na części zbiorów
w ujęciu AUC_theta. Wybór między nimi sprowadza się do akcentów: czy
priorytetem jest ogólna średnia i efektywność, czy maksymalna równość
wyników w poprzek kątów. 
W przypadku CyResNet56 trening trwał dłużej niż CyVGG19 i w większości konfiguracji
(szczególnie z log-polar na GTSRB, GTSRB_RGB i MNIST) osiągał wyższą
stabilność rotacyjną mierzoną AUC_theta. Wyjątki to LEGO w log-polar oraz
GTSRB_RGB w linear-polar, gdzie nieznacznie lepsze AUC_theta uzyskał
CyVGG19. Co ukazuje klasyczny kompromis jakość-czas: nie da się jednocześnie
maksymalizować stabilności i skracać treningu. I jak to się mówi - 
nie można zjeść ciastka i mieć go też (ang. „You can’t eat your
cake and have it too” [@kaczynski1995wp]).

\newpage

*[Rys. 8 - Porównanie per-time (avg_perf) - GTSRB]*  \
![Rys. 8: Porównanie per-time (avg_perf) GTSRB](media%2Fassets%2Fplots%2Favg_perf_GTSRB_10.png)  \
Rys. 8: Porównanie per-time (avg_perf) GTSRB  \
*[Rys. 8.1 - Porównanie per-time (avg_perf) - GTSRB-RGB]*  \
![Rys. 8.1: Porównanie per-time (avg_perf) GTSRB RGB](media%2Fassets%2Fplots%2Favg_perf_GTSRB_RGB_10.png)  \
Rys. 8.1: Porównanie per-time (avg_perf) GTSRB RGB \
*[Rys. 8.2 - Porównanie per-time (avg_perf) - LEGO]*  \
![Rys. 8.2: Porównanie per-time (avg_perf) LEGO](media%2Fassets%2Fplots%2Favg_perf_GTSRB_RGB_10.png)  \
Rys. 8.2: Porównanie per-time (avg_perf) LEGO  \
*[Rys. 8.3 - Porównanie per-time (avg_perf) - MNIST]*  \
![Rys. 8.3: Porównanie per-time (avg_perf) MNIST](media%2Fassets%2Fplots%2Favg_perf_MNIST_10.png)  \
Rys. 8.3: Porównanie per-time (avg_perf) MNIST  \
Wniosek: Efektywność per-time w przypadku modelu CyResNet56-linear jest zazwyczaj najlepsza.

## Wpływ transformacji (linear-polar vs log-polar)

W pracy stosowane są dwa warianty odwzorowania biegunowego. W trybie
**linear** promień rośnie liniowo, co daje równy krok po $\rho$ oraz
stabilną siatkę kątową. W trybie **log** promień rośnie logarytmicznie,
co zagęszcza próbki w pobliżu środka i ułatwia „składanie” dużych obrotów
wzdłuż osi $\rho$. Metryki liczone są tak samo dla obu wariantów:
**micro-accuracy** z macierzy pomyłek \texttt{confusion\_matrix.npy},
$\mathrm{AUC}_\theta$ i **worst-case** z krzywych
$\mathrm{Acc}(\Delta\theta)$ po zbinowaniu różnic kątów z wrap-around oraz
**avg\_perf** rozumiane jako $\mathrm{mean}(\mathrm{Acc})/\mathrm{train\_time}$.

**Konwencja różnic ($\Delta$).** Porównania zapisuję jako
$\Delta=\mathrm{log}-\mathrm{linear}$ w obrębie **tej samej** rodziny i
**tego samego** zbioru. Przykład:  \
`avg 0.926 → 0.954 (Δavg +0.029), AUC 0.921 → 0.952 (ΔAUC +0.031),`  \
`worst 0.916 → 0.950 (Δworst +0.034), avg_perf +0.000050)`.  \

*[Rys. 9 - Δ (log − linear) - słupki dla MNIST (np. CyResNet)]*  \
![Rys. 9 - Δ (log − linear) - słupki dla MNIST (np. CyResNet)](media%2Fassets%2Fplots%2Fdelta_log_minus_linear_MNIST_CyResNet56.png)  \
Rys. 9: Wpływ transformacji: na MNIST log-polar podnosi zarówno jakość, jak i stabilność oraz lekko poprawia per-time.  \

*[Rys. 10 - Δ (log − linear) - słupki dla GTSRB (np. CyResNet)]*  \
![Rys. 10 - Δ (log − linear) - słupki dla GTSRB (np. CyResNet)](media%2Fassets%2Fplots%2Fdelta_log_minus_linear_GTSRB_CyResNet56.png)
Rys. 10: W przypadku zbioru GTSRB model linear-polar częściej wygrywa w avg i AUC θ ; log-polar bywa korzystny punktowo.

## GTSRB (micro)

W przypadku datasetu GTSRB **linear** wygrywa, jeżeli chodzi o średnią i stabilność, użycie **log** jest korzystniejsze
per-time tylko dla CyVGG.5rtcha

**VGG19**  
avg 0.479 → 0.473 (Δavg −0.006)  
AUC 0.445 → 0.438 (ΔAUC −0.007)  
worst 0.441 → 0.434 (Δworst −0.007)

**ResNet56**  
avg 0.547 → 0.519 (Δavg −0.028)  
AUC 0.518 → 0.488 (ΔAUC −0.030)  
worst 0.514 → 0.484 (Δworst −0.030)  
avg_perf 0.001241 → 0.001070 (Δ −0.000170)

**CyVGG19**  
avg 0.854 → 0.818 (Δavg −0.036)  
AUC 0.847 → 0.809 (ΔAUC −0.038)  
worst 0.844 → 0.806 (Δworst −0.038)  
avg_perf 0.001070 → 0.001250 (Δ +0.000180)

**CyResNet56**  
avg 0.869 → 0.853 (Δavg −0.016)  
AUC 0.864 → 0.848 (ΔAUC −0.016)  
worst 0.862 → 0.844 (Δworst −0.017)  
avg_perf 0.001025 → 0.000849 (Δ −0.000176)

**Wniosek.** Priorytet **avg/AUC/worst** → **linear**. Dla **per-time**
jedyny wyjątek to **CyVGG-log**.

## GTSRB_RGB (micro)

Różnice **log** vs **linear** w rodzinie są **małe**, z lekką przewagą
**linear**. Największy skok jakości i tak daje **cykliczność**.

**VGG19**  
avg 0.504 → 0.498 (Δavg −0.007)  
AUC 0.472 → 0.464 (ΔAUC −0.007)  
worst 0.466 → 0.459 (Δworst −0.007)  
avg_perf 0.001447 → 0.001454 (Δ +0.000008)

**ResNet56**  
avg 0.591 → 0.576 (Δavg −0.015)  
AUC 0.566 → 0.549 (ΔAUC −0.017)  
worst 0.562 → 0.544 (Δworst −0.018)

**CyVGG19**  
avg 0.902 → 0.887 (Δavg −0.016)  
AUC 0.901 → 0.883 (ΔAUC −0.017)  
worst 0.900 → 0.882 (Δworst −0.018)  
avg_perf 0.001791 → 0.001692 (Δ −0.000099)

**CyResNet56**  
avg 0.898 → 0.897 (Δavg −0.002)  
AUC 0.897 → 0.895 (ΔAUC −0.002)  
worst 0.896 → 0.894 (Δworst −0.002)  
avg_perf 0.001552 → 0.001537 (Δ −0.000015)

**Wniosek.** Dla RGB **linear** jest minimalnie lepszy jakościowo i
zwykle szybszy; **log** nie daje zauważalnych zysków w micro.

## LEGO (micro)

W przyapdku datasetu LEGO wyniki pokazują ciekawą zależność. W klasycznych modelach
użycie **linear** wygrywa jeżeli chodzi o jakość. W przypadku CyResNet użycie **log** robi poprawę 
jakości i stabilności. Per-time zależy od rodziny.

**VGG19**  
avg 0.850 → 0.840 (Δavg −0.010)  
AUC 0.845 → 0.834 (ΔAUC −0.011)  
worst 0.844 → 0.833 (Δworst −0.011)  
avg_perf 0.000999 → 0.001261 (Δ +0.000262)

**ResNet56**  
avg 0.815 → 0.797 (Δavg −0.018)  
AUC 0.808 → 0.789 (ΔAUC −0.018)  
worst 0.806 → 0.788 (Δworst −0.019)  
avg_perf 0.001429 → 0.001407 (Δ −0.000022)

**CyVGG19**  
avg 0.882 → 0.876 (Δavg −0.005)  
AUC 0.879 → 0.874 (ΔAUC −0.005)  
worst 0.878 → 0.873 (Δworst −0.005)  
avg_perf 0.000961 → 0.000909 (Δ −0.000052)

**CyResNet56**  
avg 0.855 → 0.856 (Δavg +0.001)  
AUC 0.851 → 0.852 (ΔAUC +0.002)  
worst 0.848 → 0.851 (Δworst +0.003)

**Wniosek.** Jeśli celem jest **jakość/stabilność** w **CyResNet56**,
**log** ma lekki plus. W **VGG/ResNet** jakościowo lepszy jest **linear**,
choć **VGG-log** bywa szybszy.

## MNIST (micro)

W przypadku zbioru MNIST użycie **log-polar** wyraźnie pomaga, zwłaszcza 
w przypadku **CyResNet56**. Zyski są widoczne we wszystkich metrykach.

**VGG19**  
avg 0.669 → 0.669 (Δavg ~0)  
AUC 0.645 → 0.645 (ΔAUC ~0)  
worst 0.640 → 0.639 (Δworst −0.001)  
avg_perf 0.000781 → 0.000801 (Δ +0.000020)

**ResNet56**  
avg 0.705 → 0.709 (Δavg +0.004)  
AUC 0.684 → 0.687 (ΔAUC +0.004)  
worst 0.679 → 0.683 (Δworst +0.004)  
avg_perf 0.000536 → 0.000542 (Δ +0.000007)

**CyVGG19**  
avg 0.888 → 0.892 (Δavg +0.004)  
AUC 0.880 → 0.885 (ΔAUC +0.005)  
worst 0.877 → 0.881 (Δworst +0.004)  
avg_perf 0.000914 → 0.001005 (Δ +0.000091)

**CyResNet56**  
avg 0.926 → 0.954 (Δavg +0.029)  
AUC 0.921 → 0.952 (ΔAUC +0.031)  
worst 0.916 → 0.950 (Δworst +0.034)  
avg_perf 0.000517 → 0.000567 (Δ +0.000050)

**Wniosek.** 
Użycie **logpolar** daje wyraźną poprawę wyników, 
z najsilniejszym efektem uzyskanym w przypadku modelu **CyResNet**.

## Liderzy per zbiór (uśrednienia rodzin, micro)

* **GTSRB**: avg/AUC/worst **CyResNet56-linear**
  (0.8688 / 0.8640 / 0.8617), per-time **CyVGG19-log** (0.001321).
* **GTSRB_RGB**: avg/AUC/worst **CyResNet56-log**
  (0.9021 / 0.9049 / 0.9028), per-time **VGG19-log** (0.001912).
* **LEGO**: avg/AUC/worst **CyResNet56-log**
  (0.8822 / 0.8855 / 0.8823), per-time **ResNet56-linear** (0.001632).
* **MNIST**: avg/AUC/worst **CyResNet56-linear**
  (0.9549 / 0.9593 / 0.9556), per-time **ResNet56-linear** (0.001987).

### Różnice **Cy − baza** (po rodzinach, micro)

**GTSRB**  
VGG linear→Cy: Δavg +0.3745, ΔAUC +0.4022, Δworst +0.4030, Δperf −0.000107  
VGG log→Cy: Δavg +0.3450, ΔAUC +0.3706, Δworst +0.3718, Δperf +0.000242  
Res linear→Cy: Δavg +0.3216, ΔAUC +0.3460, Δworst +0.3477, Δperf −0.000200  
Res log→Cy: Δavg +0.3337, ΔAUC +0.3595, Δworst +0.3610, Δperf −0.000311

**GTSRB_RGB**  
VGG linear→Cy: Δavg +0.3867, ΔAUC +0.3977, Δworst +0.3954, Δperf −0.000171  
VGG log→Cy: Δavg +0.3907, ΔAUC +0.4015, Δworst +0.3999, Δperf −0.000194  
Res linear→Cy: Δavg +0.3959, ΔAUC +0.4103, Δworst +0.4080, Δperf −0.000243  
Res log→Cy: Δavg +0.3906, ΔAUC +0.4062, Δworst +0.4045, Δperf −0.000347

**LEGO**  
VGG linear→Cy: Δavg +0.0252, ΔAUC +0.0289, Δworst +0.0280, Δperf −0.000424  
VGG log→Cy: Δavg +0.0321, ΔAUC +0.0360, Δworst +0.0353, Δperf −0.000052  
Res linear→Cy: Δavg +0.0180, ΔAUC +0.0218, Δworst +0.0209, Δperf −0.000203  
Res log→Cy: Δavg +0.0220, ΔAUC +0.0255, Δworst +0.0247, Δperf −0.000118

**MNIST**  
VGG linear→Cy: Δavg +0.0012, ΔAUC +0.0021, Δworst +0.0017, Δperf +0.000089  
VGG log→Cy: Δavg +0.0008, ΔAUC +0.0011, Δworst +0.0008, Δperf +0.000139  
Res linear→Cy: Δavg +0.0037, ΔAUC +0.0040, Δworst +0.0037, Δperf −0.000045  
Res log→Cy: Δavg +0.0035, ΔAUC +0.0037, Δworst +0.0034, Δperf +0.000027

## GTSRB - heatmapy train–test

Poniżej zestawiam mapy train–test dla zbioru **GTSRB**. Oś pozioma to
wariant testowy (różne zakresy rotacji), oś pionowa - wariant treningowy.
Kolor odpowiada **dokładności micro**; ciemniejsze pola oznaczają lepszy
wynik. Diagonala reprezentuje zgodność rozkładów kątów (trening≈test),
natomiast kolumny odległe od diagonali pokazują, jak model **generalizuje
na rotacje, których nie widział podczas treningu**.

### VGG19 (bazowy) - linear vs log
*Rys. A: VGG19 - linear-polar*  
![VGG19 - linear-polar (GTSRB)](media/assets/heatmaps/GTSRB/heatmap_vgg19_linearpolar.png)
\newpage
*Rys. B: VGG19 - log-polar*  
![VGG19 - log-polar (GTSRB)](media/assets/heatmaps/GTSRB/heatmap_vgg19_logpolar.png)

Wariant bazowy **VGG19** koncentruje wysokie wartości w pobliżu diagonali.
Po wyjściu poza zakres kątów widocznych w treningu pojawia się stopniowe
rozjaśnienie kolumn, czyli spadek accuracy przy większych odchyleniach
kątowych. Transformacja **log-polar** łagodzi ten spadek nieco mocniej niż
**linear-polar** (bardziej równy pas w kolumnach dalszych od diagonali), ale
nie eliminuje problemu całkowicie.

### ResNet56 (bazowy) - linear vs log
*Rys. C: ResNet56 - linear-polar*  
![ResNet56 - linear-polar (GTSRB)](media/assets/heatmaps/GTSRB/heatmap_resnet56_linearpolar.png)  
\newpage
*Rys. D: ResNet56 - log-polar*  
![ResNet56 - log-polar (GTSRB)](media/assets/heatmaps/GTSRB/heatmap_resnet56_logpolar.png)  

**ResNet56** zachowuje się podobnie do VGG19: ciemna diagonala i jaśniejsze
kolumny wraz ze wzrostem różnicy kątów. Wersja **log-polar** ponownie daje
nieco stabilniejszy obraz niż **linear-polar**, szczególnie dla testów
odległych kątowo od treningu, ale nadal widać wyraźny dryf jakości.

### Modele cykliczne - CyVGG19 i CyResNet56
*Rys. E: CyVGG19 - linear-polar*  
![CyVGG19 - linear-polar (GTSRB)](media/assets/heatmaps/GTSRB/heatmap_cyvgg19_linearpolar.png)  
\newpage
*Rys. F: CyVGG19 - log-polar*  
![CyVGG19 - log-polar (GTSRB)](media/assets/heatmaps/GTSRB/heatmap_cyvgg19_logpolar.png)  
\newpage
*Rys. G: CyResNet56 - linear-polar*  
![CyResNet56 - linear-polar (GTSRB)](media/assets/heatmaps/GTSRB/heatmap_cyresnet56_linearpolar.png)  
\newpage
*Rys. H: CyResNet56 - log-polar*  
![CyResNet56 - log-polar (GTSRB)](media/assets/heatmaps/GTSRB/heatmap_cyresnet56_logpolar.png)  

Warianty **cykliczne** utrzymują wysokie wartości nie tylko na diagonali,
ale także w **kolumnach** odpowiadających testom odległym kątowo od
treningu. Mapa ma przez to bardziej wyrównany, „ciemny” profil w całym
zakresie kątów, co wskazuje na **stabilność rotacyjną**. Między
przekształceniami widać delikatną przewagę **log-polar** nad **linear-polar**
przy skrajniejszych odchyleniach, ale kluczowy efekt przynosi sama
architektura cykliczna.

### Wniosek odnośnie heatmap dla zbioru GTSRB w odcienaich szarości
Na GTSRB transformacje **log-polar** pomagają modelom bazowym,
jednak to **architektury cykliczne** (CyVGG19, CyResNet56) zapewniają
rzeczywistą odporność na zmianę orientacji. Jest to widoczne jako
utrzymanie „ciemnych” kolumn daleko od diagonali, a więc wysoka jakość
także wtedy, gdy rozkład kątów w teście znacząco różni się od treningu.
W ujęciu metryk przekłada się to na wyższe **AUC\_θ** (większe pole pod
krzywą Acc(Δθ)) i lepszy *worst-case* niż w wariantach bazowych.

## GTSRB_RGB - heatmapy train–test

Poniżej zestawiono mapy train–test dla zbioru **GTSRB_RGB**, czyli
wersji kolorowej znaków drogowych. Oś pozioma odpowiada wariantom
testowym (różne zakresy rotacji), oś pionowa - wariantom treningowym.
Ciemniejsze pola oznaczają wyższą dokładność micro-accuracy, jaśniejsze
spadek skuteczności przy większej różnicy kątów Δθ.

### VGG19 (bazowy) - linear vs log
*Rys. A (RGB): VGG19 - linear-polar*  
![VGG19 - linear-polar (GTSRB RGB)](media/assets/heatmaps/GTSRB_RGB/heatmap_vgg19_linearpolar.png)  
\newpage
*Rys. B (RGB): VGG19 - log-polar*  
![VGG19 - log-polar (GTSRB RGB)](media/assets/heatmaps/GTSRB_RGB/heatmap_vgg19_logpolar.png)  

Model **VGG19** w wersji RGB zachowuje podobny wzorzec jak dla zbioru
monochromatycznego - wyraźna diagonala oznacza dobre dopasowanie
rozkładów rotacji w treningu i teście. Wariant **log-polar** łagodzi spadek
dokładności poza tym zakresem, tworząc bardziej płynne przejście w
kolumnach testowych. Kolor poprawia stabilność przy umiarkowanych
odchyleniach (Δθ ≈ 30–60°), co wskazuje, że barwa stanowi dodatkowy
sygnał wspierający rozpoznanie kształtu.

### ResNet56 (bazowy) - linear vs log
*Rys. C (RGB): ResNet56 - linear-polar*  
![ResNet56 - linear-polar (GTSRB RGB)](media/assets/heatmaps/GTSRB_RGB/heatmap_resnet56_linearpolar.png)  
\newpage
*Rys. D (RGB): ResNet56 - log-polar*  
![ResNet56 - log-polar (GTSRB RGB)](media/assets/heatmaps/GTSRB_RGB/heatmap_resnet56_logpolar.png)  

**ResNet56** w wersji RGB wykazuje wyraźniejszą poprawę stabilności niż
wariant z obrazami monochromatycznymi. Dla obu transformacji, a
szczególnie **log-polar**, kolumny testowe utrzymują ciemniejszy profil w
szerszym zakresie kątów. Sieć wykorzystuje barwne cechy (np. czerwone
obramowania, żółte tła) jako odniesienia mniej zależne od rotacji.
Spadek jakości poza diagonalą jest łagodniejszy, lecz architektura nadal
nie jest w pełni inwariantna rotacyjnie.

### Modele cykliczne - CyVGG19 i CyResNet56
*Rys. E (RGB): CyVGG19 - linear-polar*  
![CyVGG19 - linear-polar (GTSRB RGB)](media/assets/heatmaps/GTSRB_RGB/heatmap_cyvgg19_linearpolar.png)  
\newpage
*Rys. F (RGB): CyVGG19 - log-polar*  
![CyVGG19 - log-polar (GTSRB RGB)](media/assets/heatmaps/GTSRB_RGB/heatmap_cyvgg19_logpolar.png)  
\newpage
*Rys. G (RGB): CyResNet56 - linear-polar*  
![CyResNet56 - linear-polar (GTSRB RGB)](media/assets/heatmaps/GTSRB_RGB/heatmap_cyresnet56_linearpolar.png)  
\newpage
*Rys. H (RGB): CyResNet56 - log-polar*  
![CyResNet56 - log-polar (GTSRB RGB)](media/assets/heatmaps/GTSRB_RGB/heatmap_cyresnet56_logpolar.png)  

Wersje **cykliczne** w konfiguracji RGB wykazują najwyższą odporność na
rotację. Mapa ma równomiernie ciemny profil w całym zakresie kolumn
testowych, co oznacza utrzymanie wysokiej dokładności nawet przy dużych
różnicach Δθ. Warianty **log-polar** są minimalnie bardziej wygładzone,
jednak kluczowy efekt wynika z samej architektury cyklicznej.

### Wniosek (GTSRB_RGB)
Wersja RGB zwiększa spójność między wariantami kątowymi, poprawiając
stabilność nawet w modelach klasycznych. Jednak dopiero **architektury
cykliczne** (CyVGG19, CyResNet56) zapewniają rzeczywistą inwariancję
rotacyjną - ich heatmapy są niemal jednolite, bez widocznych pasów
degradacji. Kolor wzmacnia cechy semantyczne, a cykliczność zapewnia
spójność geometryczną, co przekłada się na wyższe **AUC_θ**, mniejszy
spadek *Acc(Δθ)* oraz stabilność predykcji dla dużych rotacji.

## Wpływ koloru

Wprowadzenie informacji barwnej w zbiorze **GTSRB_RGB** poprawiło
ogólną stabilność modeli względem rotacji. Kolor dostarcza dodatkowych
wskazówek semantycznych - np. czerwone ramki, żółte tła czy niebieskie
symbole - które są rozpoznawalne niezależnie od orientacji znaku.  
Efekt ten jest szczególnie widoczny w modelach klasycznych, gdzie spadek
dokładności przy zmianie kąta jest łagodniejszy niż w wersjach
monochromatycznych. W modelach **cyklicznych** (CyVGG19, CyResNet56)
kolor działa komplementarnie, wzmacniając reprezentacje i prowadząc do
niemal pełnej inwariancji rotacyjnej.

### Podsumowanie ogólne

W przypadku zbiorów **GTSRB** i **GTSRB_RGB** w obrębie tej samej rodziny 
**linear** daje zwykle wyższą **średnią** i $\mathrm{AUC}_\theta$, zaś **log** 
sporadycznie wygrywa „per-time” (np. CyVGG na GTSRB). Dataset **LEGO** pokazuje, że:
dla **CyResNet** **log** ma delikatny plus jakości i stabilności, natomiast
w **VGG/ResNet** lepszy jakościowo pozostaje **linear**, choć **VGG-log**
bywa szybszy. W **MNIST** **log-polar** wyraźnie pomaga (najmocniej w
**CyResNet**). W praktyce, gdy priorytetem jest **odporność na duże
obroty** ($\mathrm{AUC}_\theta$, worst), warto wybierać **log**, gdy
liczy się **przepustowość i koszt** (avg\_perf), ale częściej lepiej użyć **linear**.


## Wnioski per zbiór

**GTSRB.** Średnia najwyższa dla CyResNet-linear, stabilność często najlepsza
dla CyVGG-log. Różnice worst-case wśród topowych konfiguracji są niewielkie,
co pozwala dobrać wariant pod budżet czasu lub wymagania stabilności.

**GTSRB_RGB.** Przejście do CyCNN przynosi wyraźny skok jakości i AUC_theta.
CyResNet-log zwykle prowadzi w stabilności, a szybkie klasyki (np.
VGG-log) przegrywają jakościowo z wersjami cyklicznymi.

**LEGO.** CyResNet-linear dominuje średnią i per-time, lecz CyVGG-log
potrafi minimalnie wygrać AUC_theta. To zbiór, na którym dobrze widać
różnicę akcentów między linear- a log-polar.

**MNIST.** Różnice są niewielkie, ale CyResNet-linear bywa najlepszy w
średniej, a CyVGG-linear w AUC_theta. Przy priorytecie przepustowości
rozsądnym wyborem jest ResNet-linear.

[Rys. 11 - Acc(Δθ) - panel 2×2 ]  \
![Rys. 11 - Acc(Δθ) - panel 2×2 MNIST](media%2Fassets%2Fplots%2Facc_vs_delta_MNIST_2x2.png)  \
Rys. 11 - Acc(Δθ) - panel 2×2 MNIST  \
![Rys. 11 - Acc(Δθ) - panel 2×2 GTSRB-RGB](media%2Fassets%2Fplots%2Facc_vs_delta_GTSRB_RGB_2x2.png)  \
Rys. 11 - Acc(Δθ) - panel 2×2 GTSRB-RGB  \
![Rys. 11 - Acc(Δθ) - panel 2×2 GTSRB](media%2Fassets%2Fplots%2Facc_vs_delta_GTSRB_2x2.png)  \
Rys. 11 - Acc(Δθ) - panel 2×2 GTSRB  \
![Rys. 11 - Acc(Δθ) - panel 2×2 LEGO](media%2Fassets%2Fplots%2Facc_vs_delta_LEGO_2x2.png)  \
Rys. 11 - Acc(Δθ) - panel 2×2 LEGO  \
 
Przekrój stabilności: przewaga modeli cyklicznych utrzymuje się niezależnie od zbioru.  \

## Analiza błędów i wzorce pomyłek

Macierz pomyłek to tablica o wymiarach C×C, w której wiersz odpowiada
klasie rzeczywistej, a kolumna klasie przewidzianej. Element (i, j) to
liczba (lub udział) przykładów klasy i zaklasyfikowanych jako j.
Idealny klasyfikator ma jasną przekątną (duże wartości na elementach
(i, i)) i ciemne otoczenie (niskie wartości poza przekątną).

## Co pozwalaja porównywać macierze

1. **Normalizacja wierszowa.**  
   W pracy stosowana jest normalizacja po wierszach (suma wiersza = 1).
   Dzięki temu porównujemy rozkład błędów w obrębie danej klasy, bez
   wpływu jej liczebności. Z takiej macierzy per-class accuracy to po
   prostu wartość na przekątnej w danym wierszu.

2. **Wzorce poza przekątną.**  
   - **Pary symetryczne:** podwyższone (i, j) oraz (j, i) oznaczają
     wzajemne mylenie się dwóch klas.  
   - **Pasma lub „wyspy”:** skupiska poza przekątną związane z
     podobieństwem wizualnym lub zmianą orientacji tego samego motywu.  
   - **Near-miss:** pomyłki do klasy „blisko” właściwej (np. podobny
     piktogram lub liczba), zwykle tuż obok przekątnej.

3. **Precision, recall, F1 per klasa.**  
   Z macierzy obliczyć można także recall (trafienia w wierszu), precision
   (trafienia w kolumnie) i F1. Jest to istotne na zbiorach
   niezbalansowanych, gdzie średnia top-1 może maskować systematyczne
   błędy w rzadkich etykietach.

4. **Porównania 1:1.**  
   Najbardziej przekonujące są zestawienia:
   - ten sam testowy rozkład kątów, dwa modele (np. baza vs cykliczny),
   - ten sam model, dwie transformacje (linear-polar vs log-polar).
   Dodatkowo można pokazać serię macierzy wzdłuż koszyków delta-theta
   (np. 0°, 45°, 90°), ale każda macierz pozostaje klasy×klasy.

## Co typowo widać na użytych zbiorach

- **GTSRB / GTSRB_RGB.**  
  Dominują near-missy między znakami o zbliżonym kształcie lub
  ikonografii (np. 50 vs 60). W wariantach bazowych przy testach
  rotacyjnych pojawiają się pasma błędów „odsunięte” od przekątnej
  (wrażliwość na orientację). W wersjach cyklicznych pasma te słabną,
  przekątna się rozjaśnia, a rozproszenia symetryczne maleją.

- **LEGO.**  
  Pomyłki wynikają z podobnych brył i kolorów. Modele cykliczne
  ograniczają rozlania pomiędzy klasami o zbliżonym obrysie. Efekt
  jest wyraźniejszy przy log-polar, szczególnie dla dużych zmian kąta.

- **MNIST.**  
  Trudne pary (3↔5, 4↔9, 7↔1) pozostają wyzwaniem niezależnie od
  architektury. Cykliczność nie usuwa podobieństw gestalt, ale
  redukuje pomyłki „rotacyjne” (np. 6↔9 przy dużych obrotach). Widać
  to jako czystszą przekątną i mniejsze wyspy poza przekątną.


## Konsekwencje dla wniosków

- **Potwierdzenie stabilności modeli cyklicznych.**  
  Wnioski z krzywych dokładności w funkcji delta-theta oraz z AUC
  znajdują odzwierciedlenie w macierzach: mniej rozlań poza przekątną,
  wyższa dominanta na przekątnej, słabsze pasma błędów „rotacyjnych”.

- **Rola transformacji.**  
  Log-polar lepiej wspiera przypadki dalekie od 0°, gdzie w macierzy
  najbardziej zyskują klasy historycznie „wrażliwe” na obrót. Linear-
  polar często zapewnia porównywalną jakość przy niższym koszcie,
  co w macierzach widać jako umiarkowanie czystą przekątną bez
  nadmiernych rozlań.

- **Użyteczność per-class miar.**  
  Równoległe raportowanie recall/precision/F1 per klasa ułatwia
  uchwycenie słabych punktów modelu poza samą średnią top-1 i
  pozwala planować ukierunkowane poprawki (np. dodatkowe przykłady
  lub regularizację dla trudnych klas).

## Analiza macierzy pomyłek (GTSRB / GTSRB_RGB, rotated-270-300)

Poniżej omawione są **wierszowo znormalizowane** macierze
pomyłek dla **N=43** klas w scenariuszu testowym *rotated-270-300*. Każdy
wiersz sumuje się do 1.0, więc odcień na przekątnej jest równy
**czułości (recall)** danej klasy, a wartości poza przekątną opisują
**strukturę błędów** (do jakich klas model „odpływa”). Porównywane są dwa
warianty tej samej konfiguracji danych: **ResNet56-linear** (bazowy) oraz
**CyResNet56-linear** (cykliczny).

\newpage

[Rys. 12 - „Znormalizowana macierz pomyłek (rotated-270-300) - ResNet56 linearpolar vs CyResnet56 testowane na pełnym zbiorze GTSRB_RGB(zmergowane wszystkie przypadki 0-360)]  \
![resnet56-linearpolar-rotated-270-300_test_on_merged_datasets_merged_range_full_0_360_plus_non_rotated_confusion_matrix_row_norm.png](media%2Fassets%2Fplots%2Fresnet56-linearpolar-rotated-270-300_test_on_merged_datasets_merged_range_full_0_360_plus_non_rotated_confusion_matrix_row_norm.png)  \
Rys. 12 - Znormalizowana macierz pomyłek (rotated-270-300) - ResNet56 linearpolar na pełnym zbiorze GTSRB_RGB (zmergowane wszystkie przypadki 0-360)  \
Widoczne jest rozproszenie błędów poza przekątną, szczególnie w rodzinach klas o podobnym kształcie. Przekątna osłabiona w szeregu wierszy.  \
![cyresnet56-linearpolar-rotated-270-300_test_on_merged_datasets_merged_range_full_0_360_plus_non_rotated_confusion_matrix_row_norm-copy.png](media%2Fassets%2Fplots%2Fcyresnet56-linearpolar-rotated-270-300_test_on_merged_datasets_merged_range_full_0_360_plus_non_rotated_confusion_matrix_row_norm-copy.png)  \
Rys. 12 - Znormalizowana macierz pomyłek (rotated-270-300) - CyResNet56 linearpolar na pełnym zbiorze GTSRB_RGB (zmergowane wszystkie przypadki 0-360)  \
Przekątna wyraźnie mocniejsza (wyższy recall), a wyspy poza przekątną słabsze i bardziej skupione. Spada liczba przerzutów do klas ‘symetrycznych’ kątowo.  \

Wniosek: Wersja cykliczna ma mniej rozproszonych pomyłek - ‘jaśniejsza’ przekątna oraz niemal brak pomyłek poza nią  \
Wersja cykliczna ma bardziej skupioną przekątną i mniejszą masę błędu poza przekątną, przerzuty do klas symetrycznych (±90°, ±270°) są wyraźnie słabsze. 
Row-norm CM pozwala porównać czułość per-klasa; średnia z diagonali jest wyższa w CyResNecie, co odzwierciedla obserwowane krzywe Acc(Δθ) i AUC_θ.

## Najważniejsze obserwacje

1. **Silniejsza przekątna w modelu cyklicznym.**  
   W **CyResNet56** elementy diagonalne są wyraźnie wyższe w większości
   wierszy niż w **ResNet56**, co oznacza **lepszy recall per-klasa**
   przy rotacjach 270-300°. Różnica jest szczególnie widoczna w grupie
   klas z „dołu” macierzy (ok. 36-42).

2. **Mniej rozproszonych pomyłek poza przekątną.**  
   W **ResNet56** widoczne są liczne „wyspy” błędów (podwyższone wartości
   w wielu kolumnach jednego wiersza). W **CyResNet56** te wyspy są
   **przygaszone i bardziej skupione**-zwykle pozostaje 1-2 dominujących
   „sąsiadów”, reszta zanika. To wskazuje, że reprezentacja kąta jest
   w modelu cyklicznym **bardziej informatywna**.

3. **Mniej mylenia klas o podobnej geometrii.**  
   W **ResNet56** częstsze są skupiska błędów wewnątrz „rodzin kształtów”
   (np. zbiory znaków trójkątnych/ostrzegawczych albo okrągłych/zakazów).
   W **CyResNet56** ta kumulacja jest **wyraźnie mniejsza**, co sugeruje,
   że **cykliczna konwolucja + linear-polar** lepiej separują klasy
   podobne wizualnie nawet przy dużych obrotach.

4. **Mniej „przerzutów” do klas symetrycznych.**  
   Dla rotacji 270-300° wariant bazowy częściej myli klasy z ich
   **symetrycznymi odpowiednikami kątowymi**. W modelu cyklicznym
   intensywność tych przerzutów spada, co jest spójne z wyższym
   **AUC\(_\theta\)** i łagodniejszym profilem **Acc(Δθ)**.

5. **Krótszy „długi ogon” błędów.**  
   Wiersze o dużej liczbie drobnych pomyłek (niska, poszarpana
   przekątna + wiele małych wartości poza nią) w **ResNet56** przechodzą
   w **CyResNet56** w **bardziej punktowe** profile: albo wyraźną poprawę
   przekątnej, albo skoncentrowanie pomyłek na 1-2 semantycznie najbliższych
   klasach. Z perspektywy systemowej oznacza to **niższą entropię błędu**
   i większą użyteczność predykcji top-k.

   
## Konkluzja

Na tym samym, trudnym scenariuszu rotacyjnym **CyResNet56-linear**
tworzy **bardziej „punktowe” macierze pomyłek**: przekątna jest
silniejsza, a rozproszenia poza nią mniejsze i przewidywalniejsze.
W praktyce oznacza to **większą stabilność rotacyjną** i **łatwiejszą
obsługę post-decyzyjną** (np. reguły top-k), co pozostaje w pełnej
zgodzie z metrykami globalnymi (AUC\(_\theta\), Acc(Δθ)).


## Koszt obliczeń a wybór konfiguracji

Jeśli liczy się niezawodność w całym zakresie kątów, zalecanym wyborem są
konfiguracje Cy + log-polar. Jeżeli kluczowa jest przepustowość lub
moc obliczeniowa oraz czas, lepiej sprawdzi się linear-polar, często w parze z
(Cy)ResNet-56. W zastosowaniach o zróżnicowanych wymaganiach można
rozważyć parę modeli i wybierać ścieżkę decyzyjną zależnie od profilu
zapytania.

\newpage

[Rys. 13 - „Per-time vs Avg (scatter)”]

![Rys. 13 - Per-time vs Avg (scatter) GTSRB](media%2Fassets%2Fplots%2Fperf_vs_time_scatter_GTSRB_micro.png)  \
Rys. 13 - Per-time vs Avg (scatter) GTSRB  \

\newpage

![Rys. 13 - Per-time vs Avg (scatter) GTSRB RGB](media%2Fassets%2Fplots%2Fperf_vs_time_scatter_GTSRB_RGB_micro.png)  \
Rys. 13 - Per-time vs Avg (scatter) GTSRB RGB \

\newpage

![ys. 13 - Per-time vs Avg (scatter) LEGO](media%2Fassets%2Fplots%2Fperf_vs_time_scatter_LEGO_micro.png)  \
Rys. 13 - Per-time vs Avg (scatter) LEGO  \

\newpage

![Rys. 13 - Per-time vs Avg (scatter) MNIST](media%2Fassets%2Fplots%2Fperf_vs_time_scatter_MNIST_micro.png)  \
Rys. 13 - „Per-time vs Avg (scatter)”MNIST  \
Wskazówka: „Trade-off jakość↔czas: punkty w prawym górnym rogu to konfiguracje najkorzystniejsze.”

## Rekomendacje praktyczne na podstawie wyników

Przy ograniczonym czasie trenowania i ścisłych limitach SLA dobrym
punktem startowym jest **CyResNet-linear**. Gdy zadanie wymaga wysokiej
stabilności w poprzek kątów lub mocno odbiega od rozkładu treningowego,
warto przejść na **log-polar** i rozważyć **CyVGG-log** lub
**CyResNet-log**. W kontekstach z dużą zmiennością skali i perspektywy
wariant log-polar utrzymuje równy poziom jakości i lepszy „worst-case”.

\newpage

# Strojenie hiperparametrów (Optuna) a odporność na rotacje

## Założenie i konfiguracja

Celem było sprawdzenie, czy **strojenie hiperparametrów** (learning rate,
momentum, weight decay) dla **konfiguracji `non_rotated`** potrafi
podnieść **bazową jakość** przy kątach bliskich \(0^\circ\),
oraz **przenieść się** na **stabilność rotacyjną** (AUC\(_\theta\),
worst-case) **bez** zmiany architektury i bez augmentacji rotacjami.

Do walidacji w **Optunie** wykorzystane zostały **`non_rotated`** wersje zbiorów danych. 
Wykorzystana została funkcja celu - klasyczne `val_acc` w pobliżu \(0^\circ\). 
Trening przebiegał bez rotacji, a użyta transformacja (linear-polar / log-polar) była identyczna
jak w bazowym zbiorze. Taki układ celowo „patrzy” tylko na zachowanie modelu wokół
\(0^\circ\) i nie nagradza odporności na duże \(\Delta\theta\).

Dla każdego zbioru wygenerowane zostały pliki:  \
`optuna_vs_baseline_nonrot_{GTSRB|GTSRB_RGB|LEGO|MNIST}_micro.csv`
Zawierają one pary kolumn `avg_base / avg_opt`, `AUC_base / AUC_opt`,
`worst_base / worst_opt` oraz różnice `d_avg`, `d_AUC`, `d_worst`.
W dalszej cześci tego rozdziału zostały zastosowane następujące skróty:  \
- **linear** = linear-polar  \
- **log** = log-polar  \
- **baseline** = bazowa konfiguracja.  \

## Metryki i zapis wyników

W porównaniu porównywane są dwie wersje dla danej pary (architektura, transformacja):
**Baseline (`non_rotated`)** i **Optuna (`non_rotated`)**. 
Do porównania zostały wykorzystane następujące miary:  \
- **avg** - średnia dokładność (micro) po scenariuszach testowych;  \
- **AUC$_\theta$** - pole pod krzywą $\mathrm{Acc}(\Delta\theta)$,
  znormalizowane do $[0,1]$;  \
- **worst** - minimum krzywej $\mathrm{Acc}(\Delta\theta)$
  (najtrudniejszy koszyk kątowy);  \
- **avg\_perf** - $\mathrm{mean}(\mathrm{Acc})/T_{\mathrm{train}}$,
  gdzie $T_{\mathrm{train}}$ to czas trenowania.  \ 

**Przykładowa notacja użyta tekście:**  \
avg \(0.926 \rightarrow 0.954\) (**\(\Delta\)avg +0.029**),  \
AUC\(_\theta\) \(0.921 \rightarrow 0.952\) (**\(\Delta\)AUC +0.031**),  \
worst \(0.916 \rightarrow 0.950\) (**\(\Delta\)worst +0.034**).  \

### Obraz globalny

Wynik jest spójny między zbiorami: zmiany **avg** są na ogół niewielkie,
zwykle \(|\Delta| \le 0.01\). Wskaźniki **AUC\(_\theta\)** i **worst**
przeważnie nie rosną, ponieważ przy walidacji ustawionej na \(0^\circ\)
brakuje presji selekcyjnej na stabilność dla dużych \(\Delta\theta\).
Zdarzają się lokalne plusy (pojedyncze pary model-transformacja), jednak
w ujęciu przekrojowym dominują wyniki neutralne lub lekko ujemne zarówno
w **AUC\(_\theta\)**, jak i w **worst**.

## Wyniki: Optuna kontra Baseline 

#### Optuna vs baseline - nonrot_GTSRB_RGB (micro)

| arch | act | avg_opt | AUC_opt | worst_opt | avg_base | AUC_base | worst_base |
|---|---|---|---|---|---|---|---|
| cyresnet56 | linearpolar | 0.872 | 0.861 | 0.831 | 0.924 | 0.923 | 0.919 |
| cyresnet56 | logpolar | 0.868 | 0.857 | 0.822 | 0.930 | 0.928 | 0.923 |
| cyvgg19 | linearpolar | 0.839 | 0.835 | 0.766 | 0.939 | 0.939 | 0.933 |
| cyvgg19 | logpolar | 0.790 | 0.791 | 0.688 | 0.938 | 0.938 | 0.931 |
| resnet56 | linearpolar | 0.588 | 0.582 | 0.389 | 0.890 | 0.887 | 0.869 |
| resnet56 | logpolar | 0.534 | 0.524 | 0.323 | 0.893 | 0.890 | 0.869 |
| vgg19 | linearpolar | 0.360 | 0.364 | 0.180 | 0.883 | 0.881 | 0.863 |
| vgg19 | logpolar | 0.335 | 0.338 | 0.141 | 0.882 | 0.881 | 0.862 |

| arch | act | d_avg | d_AUC | d_worst |
|---|---|---|---|---|
| cyresnet56 | linearpolar | -0.052 | -0.062 | -0.088 |
| cyresnet56 | logpolar | -0.061 | -0.071 | -0.101 |
| cyvgg19 | linearpolar | -0.100 | -0.104 | -0.167 |
| cyvgg19 | logpolar | -0.148 | -0.147 | -0.242 |
| resnet56 | linearpolar | -0.301 | -0.305 | -0.481 |
| resnet56 | logpolar | -0.359 | -0.366 | -0.546 |
| vgg19 | linearpolar | -0.523 | -0.517 | -0.684 |
| vgg19 | logpolar | -0.547 | -0.542 | -0.722 |

> **GTSRB_RGB - CyResNet-log**  
> `avg 0.930 → 0.868` (**$\Delta$avg -0.061**)  
> `AUC 0.928 → 0.857` (**$\Delta$AUC -0.071**)  
> `worst 0.923 → 0.822` (**$\Delta$worst -0.101**)

### Optuna vs baseline - nonrot_GTSRB (micro)

| arch | act | avg_opt | AUC_opt | worst_opt | avg_base | AUC_base | worst_base |
|---|---|---|---|---|---|---|---|
| cyresnet56 | linearpolar | 0.844 | 0.838 | 0.793 | 0.937 | 0.935 | 0.927 |
| cyresnet56 | logpolar | 0.790 | 0.770 | 0.680 | 0.927 | 0.924 | 0.911 |
| cyvgg19 | linearpolar | 0.690 | 0.692 | 0.573 | 0.923 | 0.921 | 0.904 |
| cyvgg19 | logpolar | 0.639 | 0.648 | 0.473 | 0.916 | 0.915 | 0.898 |
| resnet56 | linearpolar | 0.351 | 0.345 | 0.212 | 0.890 | 0.887 | 0.870 |
| resnet56 | logpolar | 0.445 | 0.437 | 0.264 | 0.878 | 0.875 | 0.855 |
| vgg19 | linearpolar | 0.299 | 0.304 | 0.142 | 0.869 | 0.867 | 0.846 |
| vgg19 | logpolar | 0.291 | 0.296 | 0.137 | 0.862 | 0.861 | 0.840 |

| arch | act | d_avg | d_AUC | d_worst |
|---|---|---|---|---|
| cyresnet56 | linearpolar | -0.093 | -0.096 | -0.135 |
| cyresnet56 | logpolar | -0.137 | -0.155 | -0.231 |
| cyvgg19 | linearpolar | -0.233 | -0.229 | -0.331 |
| cyvgg19 | logpolar | -0.276 | -0.267 | -0.425 |
| resnet56 | linearpolar | -0.539 | -0.542 | -0.658 |
| resnet56 | logpolar | -0.433 | -0.438 | -0.590 |
| vgg19 | linearpolar | -0.570 | -0.563 | -0.703 |
| vgg19 | logpolar | -0.572 | -0.565 | -0.703 |

> **GTSRB - ResNet-linear**  
> `avg 0.890 → 0.351` (**$\Delta$avg -0.539**)  
> `AUC 0.887 → 0.345` (**$\Delta$AUC -0.542**)  
> `worst 0.870 → 0.212` (**$\Delta$worst -0.658**)

### Optuna vs baseline - nonrot_LEGO (micro)

| arch | act | avg_opt | AUC_opt | worst_opt | avg_base | AUC_base | worst_base |
|---|---|---|---|---|---|---|---|
| cyresnet56 | linearpolar | 0.660 | 0.638 | 0.562 | 0.871 | 0.867 | 0.853 |
| cyresnet56 | logpolar | 0.728 | 0.718 | 0.671 | 0.878 | 0.876 | 0.868 |
| cyvgg19 | linearpolar | 0.789 | 0.783 | 0.753 | 0.905 | 0.905 | 0.901 |
| cyvgg19 | logpolar | 0.770 | 0.768 | 0.755 | 0.903 | 0.902 | 0.899 |
| resnet56 | linearpolar | 0.513 | 0.494 | 0.452 | 0.878 | 0.875 | 0.864 |
| resnet56 | logpolar | 0.609 | 0.594 | 0.541 | 0.868 | 0.866 | 0.857 |
| vgg19 | linearpolar | 0.706 | 0.702 | 0.660 | 0.903 | 0.903 | 0.897 |
| vgg19 | logpolar | 0.581 | 0.576 | 0.559 | 0.898 | 0.898 | 0.893 |

| arch | act | d_avg | d_AUC | d_worst |
|---|---|---|---|---|
| cyresnet56 | linearpolar | -0.212 | -0.229 | -0.291 |
| cyresnet56 | logpolar | -0.150 | -0.158 | -0.197 |
| cyvgg19 | linearpolar | -0.116 | -0.122 | -0.148 |
| cyvgg19 | logpolar | -0.133 | -0.134 | -0.145 |
| resnet56 | linearpolar | -0.365 | -0.381 | -0.412 |
| resnet56 | logpolar | -0.259 | -0.272 | -0.316 |
| vgg19 | linearpolar | -0.197 | -0.201 | -0.236 |
| vgg19 | logpolar | -0.317 | -0.322 | -0.335 |

> **LEGO - CyVGG-log**  
> `avg 0.903 → 0.770` (**$\Delta$avg -0.133**)  
> `AUC 0.902 → 0.768` (**$\Delta$AUC -0.134**)  
> `worst 0.899 → 0.755` (**$\Delta$worst -0.145**)

### Optuna vs baseline - nonrot_MNIST (micro)

| arch | act | avg_opt | AUC_opt | worst_opt | avg_base | AUC_base | worst_base |
|---|---|---|---|---|---|---|---|
| cyresnet56 | linearpolar | 0.866 | 0.866 | 0.694 | 0.958 | 0.957 | 0.925 |
| cyresnet56 | logpolar | 0.921 | 0.919 | 0.851 | 0.975 | 0.975 | 0.965 |
| cyvgg19 | linearpolar | 0.750 | 0.751 | 0.614 | 0.949 | 0.948 | 0.927 |
| cyvgg19 | logpolar | 0.782 | 0.782 | 0.633 | 0.953 | 0.953 | 0.930 |
| resnet56 | linearpolar | 0.604 | 0.604 | 0.428 | 0.933 | 0.932 | 0.902 |
| resnet56 | logpolar | 0.607 | 0.607 | 0.396 | 0.941 | 0.940 | 0.923 |
| vgg19 | linearpolar | 0.512 | 0.513 | 0.265 | 0.921 | 0.920 | 0.894 |
| vgg19 | logpolar | 0.495 | 0.495 | 0.255 | 0.920 | 0.920 | 0.890 |

| arch | act | d_avg | d_AUC | d_worst |
|---|---|---|---|---|
| cyresnet56 | linearpolar | -0.092 | -0.091 | -0.230 |
| cyresnet56 | logpolar | -0.055 | -0.056 | -0.114 |
| cyvgg19 | linearpolar | -0.198 | -0.197 | -0.313 |
| cyvgg19 | logpolar | -0.171 | -0.171 | -0.297 |
| resnet56 | linearpolar | -0.329 | -0.329 | -0.473 |
| resnet56 | logpolar | -0.334 | -0.334 | -0.528 |
| vgg19 | linearpolar | -0.408 | -0.408 | -0.630 |
| vgg19 | logpolar | -0.426 | -0.425 | -0.636 |

> **MNIST - ResNet-linear**  
> `avg 0.933 → 0.604` (**$\Delta$avg -0.329**)  
> `AUC 0.932 → 0.604` (**$\Delta$AUC -0.329**)  
> `worst 0.902 → 0.428` (**$\Delta$worst -0.473**)


*(Komentarz: „walidacja `non_rotated` nie podnosi
AUC\(_\theta\)/worst; zmiana avg kosmetyczna”).*

## Interpretacja: dlaczego tak wyszło

1. **Niedopasowany cel walidacji.** Walidacja `non_rotated` faworyzuje
   konfiguracje pod \(\Delta\theta \approx 0^\circ\). AUC\(_\theta\) i
   worst zależą od zachowania przy **dużych** \(\Delta\theta\), którego
   Optuna **nie mierzy**.
2. **Architektura dominuje nad LR/WD.** Odporność na obrót wynika
   przede wszystkim z **własności modelu** (CyCNN, oś orientacji,
   transformacje polarne). Hiperparametry regulują tempo/gładkość
   uczenia, ale nie wprowadzają **ekwiwariancji**.
3. **Budżet i harmonogram.** Krótki trening lub konserwatywny scheduler
   zmniejsza „rozdzielczość” selekcji; łatwo przestroić się pod szybki
   wzrost w okolicy \(0^\circ\).
4. **Regularizacja nie pod AUC.** Jeśli przestrzeń obejmuje tylko LR,
   WD i momentum, trudno poprawić **worst/AUC\(_\theta\)**. Pomogłyby
   zabiegi wzmacniające uogólnianie (label smoothing, dropout schedule,
   mixup/cutmix).
5. **Bias prunera.** Wczesne zatrzymywanie oparte o `val_acc` przy
   \(0^\circ\) premiuje konfiguracje „szybkiego startu” kosztem
   globalnej stabilności.

## Co to znaczy dla wniosków w pracy

Brak systematycznego zysku w **AUC\(_\theta\)** / **worst** przy
**val = `non_rotated`** potwierdza, że o odporności decyduje
**architektura + transformacja**, a nie samo dostrajanie LR/WD pod
\(0^\circ\). To **nie** dowód na „nieskuteczność Optuny”, tylko sygnał,
że **cel walidacji** był **niespójny** z badaną własnością
(stabilnością rotacyjną).

# Optuna z walidacją rotacyjną

## Motywacja dodatkowego testu

Chciałem sprawdzić, czy da się „wyciągnąć” większą odporność na obrót
samą zmianą **tego, jak wybieramy najlepszy checkpoint**. Trening
pozostaje *non_rotated* (bez obrotów w danych), ale walidacja i kryterium
wyboru modelu patrzą już na **zachowanie względem kątów**. Innymi słowy:
czy mądrzejsza walidacja wystarczy, żeby poprawić stabilność, **bez**
modyfikacji architektury i bez dokładania rotacyjnego augmentu.

## Zmienione ustawienia względem standardowej optuny

Rozdzieliłem ścieżki danych: `train_dir` to czyste *non_rotated*, a
`val_dir` zawiera **miks kątów** (np. `non_rotated`, `rotated-XX`,
`range_*`, `full_0_360`). Na tej walidacji liczę krzywą
$\mathrm{Acc}(\Delta\theta)$ (koszyki co $15^\circ$, wrap-around) i
z niej biorę **$\mathrm{AUC}_\theta$**, **worst** oraz **avg**. W Optunie
ustawiam **cel walidacji na $\mathrm{AUC}_\theta$**, żeby selekcjonować
checkpointy, które lepiej trzymają poziom w całym zakresie kątów. Sam
**trening** pozostaje **taki jak był**.


### GTSRB: Optuna-A vs baseline
(oba: trening non_rotated; ta sama architektura i transformacja)
Wynik ogólny. W tej próbie rotation-aware walidacja nie podniosła
jakości. Dla każdej z ośmiu konfiguracji spadły avg, AUCθ i worst.

**Szybki bilans ($\Delta$):**

* **CyResNet56 · linear**  
  avg $0.9371 \to 0.8252$ (**$\Delta$avg $-0.1118$**),  
  $\mathrm{AUC}_\theta$ $0.9345 \to 0.8163$ (**$\Delta$AUC $-0.1183$**),  
  worst $0.9273 \to 0.7543$ (**$\Delta$worst $-0.1731$**)

* **CyResNet56 · log**  
  avg $0.9268 \to 0.6842$ (**$\Delta$avg $-0.2425$**),  
  $\mathrm{AUC}_\theta$ $0.9243 \to 0.6527$ (**$\Delta$AUC $-0.2715$**),  
  worst $0.9113 \to 0.5044$ (**$\Delta$worst $-0.4070$**)

* **CyVGG19 · linear**  
  avg $0.9229 \to 0.6749$ (**$\Delta$avg $-0.2480$**),  
  $\mathrm{AUC}_\theta$ $0.9209 \to 0.6757$ (**$\Delta$AUC $-0.2452$**),  
  worst $0.9039 \to 0.5402$ (**$\Delta$worst $-0.3637$**)

* **CyVGG19 · log**  
  avg $0.9156 \to 0.6533$ (**$\Delta$avg $-0.2623$**),  
  $\mathrm{AUC}_\theta$ $0.9145 \to 0.6530$ (**$\Delta$AUC $-0.2615$**),  
  worst $0.8977 \to 0.5007$ (**$\Delta$worst $-0.3970$**)

* **ResNet56 · linear**  
  avg $0.8901 \to 0.4279$ (**$\Delta$avg $-0.4622$**),  
  $\mathrm{AUC}_\theta$ $0.8868 \to 0.4266$ (**$\Delta$AUC $-0.4602$**),  
  worst $0.8696 \to 0.2416$ (**$\Delta$worst $-0.6280$**)

* **ResNet56 · log**  
  avg $0.8777 \to 0.4200$ (**$\Delta$avg $-0.4577$**),  
  $\mathrm{AUC}_\theta$ $0.8748 \to 0.4101$ (**$\Delta$AUC $-0.4646$**),  
  worst $0.8548 \to 0.2645$ (**$\Delta$worst $-0.5904$**)

* **VGG19 · linear**  
  avg $0.8686 \to 0.2932$ (**$\Delta$avg $-0.5755$**),  
  $\mathrm{AUC}_\theta$ $0.8669 \to 0.2988$ (**$\Delta$AUC $-0.5681$**),  
  worst $0.8458 \to 0.1339$ (**$\Delta$worst $-0.7119$**)

* **VGG19 · log**  
  avg $0.8623 \to 0.2892$ (**$\Delta$avg $-0.5731$**),  
  $\mathrm{AUC}_\theta$ $0.8614 \to 0.2956$ (**$\Delta$AUC $-0.5659$**),  
  worst $0.8403 \to 0.1303$ (**$\Delta$worst $-0.7100$**)


## Wyniki praktyczne dla zbioru (GTSRB)

Krótko i uczciwie tuning z użyciem optuny i zmienionego kryterium walidacji **nie zadziałało**. 
Dla wszystkich ośmiu konfiguracji (VGG/ResNet i ich warianty cykliczne, linear/log)
**spadły** trzy kluczowe miary: średnia dokładność, ($\mathrm{AUC}_\theta$)
i „worst-case”. Przykładowo, **CyResNet56·linear** z
($\mathrm{avg}\approx0{.}94$) zjechał do ok. (0{.}83), a „worst”
spadł wyraźnie poniżej poprzedniego minimum. Analogiczny obraz pojawił
się dla pozostałych par, łącznie z modelami bazowymi.

## Dlaczego tak się stało

Najprostsze wyjaśnienie to **parytet budżetu** i **materiał do selekcji**.
Baseline’y, z którymi się porównuję, to agregaty z dłuższych i bogatszych
treningów (wiele scenariuszy, więcej epok). Wariant z Optuną miał
**krótsze przebiegi**, więc mimo „sprytnego” celu walidacji selekcjonował
spośród **słabiej wygrzanych** checkpointów. Druga sprawa: **zmieniłem
tylko walidację, a nie ekspozycję** na kąty w samym uczeniu. To pomaga
wybrać lepszy punkt *spośród tego, co powstało*, ale nie stworzy
od zera cech, które normalnie budujemy architekturą (CyCNN, oś
orientacji) albo augmentacją rotacją. Dochodzi jeszcze ryzyko
**niedopasowania rozkładu walidacji do testów**: jeśli miks kątów na
walidacji różni się od tego w docelowych scenariuszach, to
($\mathrm{AUC}*\theta$) z walidacji słabiej koreluje z
($\mathrm{AUC}*\theta$) w testach. Na koniec: **worst-case** jest
najbardziej wrażliwy na „niedogrzanie”. Krótkie treningi po prostu nie
dowożą dojrzałych reprezentacji, które trzymają poziom także dla
najtrudniejszych ($\Delta\theta$).

## Skutki zmiany kryterium walidacji

Zmiana kryterium walidacji **porządkuje selekcję checkpointu**, ale **nie
zastąpi** dłuższego, bogatszego uczenia i **nie skompensuje** różnic
architektonicznych. **Odporność na rotację** w tej pracy wynika przede
wszystkim z **CyCNN** (oś orientacji, cykliczny padding) oraz z
**przekształceń polarnych**. Samo strojenie **LR/momentum/WD pod
($\mathrm{AUC}_\theta$)**, przy krótkim budżecie i bez rotacji w treningu,
**nie tworzy ekwiwariancji** - co najwyżej wybiera trochę lepszy punkt
na istniejącej, dość płaskiej krzywej.

## Dalszy rozwój

Żeby taki wariant miał sens, trzeba **wyrównać budżet** z baseline’ami
(epoki, batch, scheduler), dopasować **walidację kątową** do tego, co
sprawdzamy w testach (te same koszyki $\Delta\theta$), a cel Optuny
ustawić na **$\mathrm{AUC}_\theta$** albo **mieszankę**
$0.7 \cdot \mathrm{avg} + 0.3 \cdot \mathrm{AUC}_\theta$ z **worst** jako
tie-breaker. Delikatne **regularizacje** (np. label smoothing, dropout)
i łagodniejsza dynamika uczenia (dłuższy warm-up, cosine LR, ewentualnie
zamrożone BN) często poprawiają dół krzywej, czyli to, co najbardziej
nas boli w kontekście rotacji.

**Jedno zdanie na koniec.**
Rotation-aware validation to dobry pomysł na **lepszy wybór checkpointu**,
ale to **nie magiczna różdżka** - bez czasu i bez wsparcia architektury nie
podniesie stabilności tak, jak robią to **CyCNN** i **log/linear-polar**.
Szczególnie, gdy wykorzystamy do treningu dane w których obrazy były obrócone o jakiś kąt.

## Wniosek syntetyczny

Przy **walidacji `non_rotated`** strojenie Optuną **nie** przekłada się
na lepszą **stabilność rotacyjną**. Hiperparametry potrafią skorygować
bazę w okolicach \(0^\circ\), lecz **AUC\(_\theta\)** i **worst**
pozostają praktycznie bez zmian. Jeśli celem jest odporność na obrót,
**architektura cykliczna** i **transformacje polarne** są kluczowe, a
proces strojenia musi tę własność **jawnie** mierzyć już na etapie
walidacji.

# Podsumowanie oraz wnioski

Architektury cykliczne są prostym, a skutecznym sposobenm podniesieniem
dokładnosci klasycznych CNN pod kątem rotacji. W średniej jakości zyskują na
wszystkich zbiorach, w stabilności wyraźnie wygrywają na danych
bogatszych wizualnie, a dzięki dwóm wariantom odwzorowania pozwalają
zbalansować **odporność** na rotacje i **efektywność**. Wybór między linear-polar, a
log-polar to wybór akcentów, a różnice między **CyVGG** i **CyResNet**
sprowadzają się do kompromisu między **stabilnością**, a **kosztem**.
W praktyce łatwo dopasować konfigurację do priorytetu wdrożeniowego bez
przebudowy głębszej części modeli, dzięki wykorzystaniu wartstw Cy w istniejących modelach CNN.
Skuteczność i generalizacje modeli CyCNN można oczywiście zwiększyć wykorzystując
do terningu i validacji pre-rotowane zbiory danych dzięki czemu można, użyć potem danego modelu
CyCNN do przypadków, których model jeszcze nie widział w sposób łatwieszy i uzyskując wyższą skuteczność
niż w wypadku użycia modeli CyCNN uczonych na niezrotowanych danych, aczkowiek nawet uczony na niezrotowanych 
danych model CyCNN (czy to CyVGG czy Resnet) jest skuteczniejszy dla zrotowanych danych niż modele CNN 
nawet po nauce z rotacją, gdyż nie posiadają one odporności na rotacje obrazów.

\begin{table}[ht]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccc}
\hline
Zbiór & Lider średniej (avg) & Lider AUC\(_\theta\) & Lider avg\_perf \\
\hline
GTSRB      & CyResNet56-linear (0.8688)  & CyResNet56-linear (0.8640)  & CyVGG19-log (0.001250) \\
GTSRB\_RGB & CyVGG19-linear (0.9025)     & CyVGG19-linear (0.9006)     & CyVGG19-linear (0.001791) \\
LEGO       & CyVGG19-linear (0.8816)     & CyVGG19-linear (0.8789)     & ResNet56-linear (0.001429) \\
MNIST      & CyResNet56-log (0.9544)     & CyResNet56-log (0.9520)     & CyVGG19-log (0.001005) \\
\hline
\end{tabular}%
}
\caption{Liderzy średniej jakości, stabilności (AUC\(_\theta\)) i efektywności
per-time dla metryki micro; porównanie rodzin i transformacji.}
\end{table}

Źródło danych: dane z results/exports/<DATASET>/micro/*.csv skonsolidowane w family_summary_<DATASET>_micro.csv.
Interpretacja: kolumny wskazują najlepszą rodzinę/transformację dla średniej jakości (avg), stabilności rotacyjnej 
(AUC_θ) i efektywności „na jednostkę czasu” (avg_perf).

\newpage

## Skuteczność rotacyjnych architektur
[WSTAWIĆ: Rys. A + B + C] + akapit „co widać i dlaczego”

Na danych o bogatszej strukturze (GTSRB_RGB) modele cykliczne utrzymują wysoką Acc(Δθ) w całym zakresie 
[0°, 180°] (rys. A), co przekłada się na najwyższe AUC_θ w rankingach (rys. B). 
Heatmapy train-test (rys. C) potwierdzają lepszą generalizację poza rozkład treningowy - „ciepłe” kolumny 
pozostają również dla odległych kątów.

## Wnioski z automatyzacji i systematyzacji ewaluacji
[WSTAWIĆ: Rys. D + E] + akapit o pipeline i porównywalności metryk

Spójny pipeline (eksporty rodzin, Acc(Δθ), AUC_θ, worst, per-time) umożliwia porównania 
między zbiorami i modelami. Panel 2×2 (rys. D) pokazuje, że przewaga cyklicznych utrzymuje się 
niezależnie od zbioru. Z kolei scatter (rys. E) odsłania kompromis jakość↔czas i ułatwia wybór 
konfiguracji wdrożeniowej.

## Propozycje dalszych badań
[WSTAWIĆ: Rys. F + G (+ H opcjonalnie)] + akapit „kiedy log vs linear, jak stroić”

Wykresy różnic (rys. F-G) pomagają rozstrzygnąć czego lepiej użyć „log vs linear” zależnie od zastosowania: log-polar 
podnosi worst/AUC_θ na zbiorach prostszych, linear częściej wygrywa średnią i per-time na GTSRB. 
Dalsze prace: walidacja rotation-aware w strojeniach, dopięcie budżetów treningowych i lekkie 
regularizacje ukierunkowane na worst. Dobrym pomysłem jest też próba zmiejszenia modelu oraz FLOPs i zapotrzebowania na VRAM.

\newpage

# Aneks

## Listingi kodów

## Dodatkowe wykresy, tablice wyników

\newpage


