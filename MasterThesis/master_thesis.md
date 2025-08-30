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

Obrazy otaczają nas z każdej strony: od zdjęć ze smartfonów, zdjęcia
satelitarne, przez monitoring miejski, katalogi produktów i systemy
kontroli jakości na liniach produkcyjnych, po systemy wspomagania jazdy.
Choć współczesne modele rozpoznawania obrazu radzą sobie bardzo dobrze,
w praktyce bywają wrażliwe na pozornie drobne zmiany takie jak
obrócenie obiektu o kilkanaście stopni czy niewielki przechył kamery.
To, co dla człowieka jest naturalne i natychmiast rozpoznawalne (znak
drogowy pod kątem, cyfra obrócona na kartce), dla klasycznej
konwolucyjnej sieci neuronowej bywa problemem. Największy problem to brak
naturalnej inwariantności względem rotacji: standardowe CNN-y „z
definicji” lepiej radzą sobie z przesunięciami niż z obrotami
[@goodfellow2016deep; @dumoulin2016guide].

W ostatnich latach pojawiło się kilka dróg domknięcia tej luki. Jedna to
poszerzanie danych o zrotowane przykłady, które poprawiają odporność, ale
wydłużają trening i nie gwarantują uogólnienia na wszystkie kąty. Druga to
architektury z wbudowaną geometrią: sieci grupowo równoważne (G-CNN,
E(2)-equivariant) [@cohen2016group; @kim2020cycnn], sieci cykliczne
(CyCNN; w szczególności **CyVGG** i **CyResNet**) operujące na wielu
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
co skróciło czas trenowania i umożliwiło szeroki przegląd eksperymentów;
środowisko uruchomieniowe ustandaryzowano z użyciem **Dockera** dla
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
**NVIDIA GeForce RTX 3070 Ti 8 GB** oraz **RTX 3060 12 GB**, co
umożliwiło przyspieszenie trenowania i testowania. Otrzymane wyniki zostały
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
  **MNIST** (odręczne cyfry, 28x28 → 32x32, grayscale) [@lecun1998mnist],
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

- Odporność na silne zakłócenia (szum, okluzje) - poza zakresem; skupiono
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

## Wprowadzenie do sieci konwolucyjnych (CNN)

Sieci konwolucyjne (CNN) zostały zaprojektowane do pracy na danych o strukturze
siatkowej lub macierzowej, takich jak obrazy dwuwymiarowe (2D). Ich kluczowe cechy to
**lokalne receptywne pola**, **współdzielone wagi** oraz **operacja splotu**. Dzięki
temu możliwe jest skalowanie modeli na duże obrazy oraz lepsze uogólnianie niż w
przypadku sieci posiadającej pełne połączenia.

Zamiast analizować cały obraz jednocześnie, CNN wykorzystują mały filtr, który
przesuwa się po lokalnych fragmentach obrazów. W ten sposób uczą się prostych
detektorów (np. krawędzi, tekstur, kształtów), a w wyższych warstwach -
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

**Kształty.**  
Wejście: $X \in \mathbb{R}^{C_{\text{in}}\times H\times W}$.  
Zestaw jąder: $K \in \mathbb{R}^{C_{\text{out}}\times C_{\text{in}}\times k\times k}$.  
Wyjście: $Y \in \mathbb{R}^{C_{\text{out}}\times H'\times W'}$.

**Definicja (pojedynczy kanał wyjściowy $c$):**
$$
Y_c(u,v)=\sum_{i=1}^{C_{\text{in}}}\sum_{a,b}
K_{c,i}(a,b)\,X_i(u-a,\;v-b).
$$

W praktyce większość frameworków oblicza **korelację krzyżową** (bez
odwracania jądra), mimo że w API funkcja nazywana jest `conv` [@dumoulin2016guide].  
Nie ma to jednak końcowo znaczenia dla procesu uczenia, bo sieć i tak dobierze
właściwe wagi.

#### Stride, padding, rozmiary

Parametry „geometrii” warstwy:

- **padding** $p$ - ile pikseli dodajemy na brzegach;
- **stride** $s$ - co ile pikseli przesuwamy okno;
- **dylacja** $d$ - „rozciąga” jądro poprzez wstawienie przerw między próbkami.

#### Uwaga o „same/valid/stride” a ekwiwariancji\

Dokładna ekwiwariancja translacyjna zachodzi przy splocie bez zmiany rozmiaru.
W praktyce **padding „same”**, **stride $>1$** i **pooling** wprowadzają drobne
odchylenia (aliasing na siatce próbkowania), co obniża „idealność”
ekwiwariancji - efekt ten jest znany i opisywany w literaturze
[@dumoulin2016guide; @azulay2019small].

**Rozmiar wyjścia** (dla jądra $k\times k$):
$$
H'=\Big\lfloor \frac{H+2p-d\,(k-1)-1}{s}\Big\rfloor+1,\qquad
W'=\Big\lfloor \frac{W+2p-d\,(k-1)-1}{s}\Big\rfloor+1.
$$

**Typowe ustawienia.**  
- *valid*: $p=0$ - mapy cech maleją;  
- *same* (dla $s=1$): $p=\lfloor k/2\rfloor$ - $H'=H$, $W'=W$;  
- *stride $>1$*: wbudowane **podpróbkowanie** (mniej obliczeń, mniejsza
  rozdzielczość);  
- *dylacja $>1$*: większe **efektywne** pole widzenia bez nowych parametrów
  (częste w detekcji/segmentacji) [@dumoulin2016guide].

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

#### Pooling po orientacjach - szczegóły praktyczne\

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

### Triki architektoniczne

- **Bazy:** VGG-19 (bloki $3{\times}3$) i ResNet-56 (wariant CIFAR, bloki $3{\times}3$).
- **Wersje cykliczne:** podmiana `Conv` → `CyConv`, dodanie osi **orientacja** i
  **cykliczny padding** po $\varphi$; pozostałe elementy (głębokość, liczba
  kanałów) dobrano tak, aby utrzymać *porównywalny budżet parametrów/FLOPs*
  względem baz.
- **Nie wprowadzano zmian niezwiązanych z rotacją, aby nie
  mieszać efektów** [@kim2020cycnn].

### Ekwiwariancja translacyjna (kontrast do rotacyjnej)

Dla przesunięcia $\mathcal{T}_t$ i splotu zachodzi:
$$
\mathcal{T}_t(X) * K \;=\; \mathcal{T}_t\!\big(X*K\big),
$$
co tłumaczy, dlaczego klasyczne CNN dobrze radzą sobie z translacją
[@dumoulin2016guide]. Brak analogicznego mechanizmu dla rotacji motywuje
użycie przekształceń polarnych i/lub modeli cyklicznych w dalszej części.

#### Ekwiwariancja rotacyjna w dyskretnej grupie $C_n$\

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

W praktyce:  
- **translacja:** uśrednianie / podpróbkowanie (GAP, stride);  
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

## Problemy z rotacyjną inwariancją w klasycznych CNN

- **Kierunkowość filtrów.** Pojedynczy kernel jest wrażliwy głównie na jedną
  orientację, gbyż bez dodatkowych mechanizmów sieć „gubi” obroty.
- **Augmentacja nie domyka całości.** Rotacje pomagają, ale wydłużają trening i
  zostawiają „dziury” między kątami (przy małym kroku i ograniczonym budżecie).
- **Aliasing / interpolacja.** Obracanie dyskretnych obrazów wprowadza artefakty
  i szum [@azulay2019small].
- **Krawędzie i padding.** „same/zero” łamie symetrię przy brzegach przez co odpowiedzi
  nie są idealnie ekwiwariantne.
- **Brak osi orientacji.** W standardowych CNN nie zapisuje się informacji o
  kącie, pod którym wykryto aktywację - dlatego trudno później uzyskać
  rozpoznawanie niezależne od obrotu.


## Przegląd literatury (E(2)-equivariant, CyCNN)

**CyCNN (podejście użyte w pracy).** Obraz przemapowano do $(\rho,\varphi)$ i
przetwarzany jest warstwami cylindrycznymi (**CyConv**) z **cyklicznym
paddingiem** wzdłuż $\varphi$. Dla każdego filtra wykorzystano $n$ orientacji
(grupa $C_n$). Obrót wejścia z $C_n$ powoduje **cykliczne przesunięcie** po osi
orientacji (**ekwiwariancja**), a **pooling po orientacjach** zapewnia
**inwariancję**. W badaniach użyte zostały modele **CyVGG** i **CyResNet**
[@kim2020cycnn].

**E(2)-equivariant / steerable CNNs (użyte jako kontekst).** Sploty grupowe i jądra
sterowalne umożliwiają ekwiwariancję względem translacji i rotacji (także dla
kątów ciągłych) w grupie $\mathrm{E}(2)$. Wymaga to projektowania jąder zgodnie
z reprezentacjami grupy i zwykle wiąże się z większym kosztem jeżeli chodzi o moc obliczeniową. 
W tej pracy traktujemy je jako tło teoretyczne
[@cohen2016group; @weiler2019general; @cohen2019homogeneous].

\newpage

# Opis zbiorów danych

## MNIST (cyfry odręczne)

Zbiór MNIST to klasyczny benchmark rozpoznawania cyfr 0–9 [@lecun1998gradient]. Zbiór
zawiera **60 000** próbek uczących i **10 000** testowych, obrazy mają rodzielczość
**28×28**, w skali szarości, piksele posiadające odcień szarości w zakresie [0, 255] 
(w pracy są one normalizowane do przedziału [0, 1] i dalej standaryzowane) 
[@mnist_web]. Szczegóły formatu i plików są 
dostępne na oficjalnej stronie MNIST [@mnist_web].

**Przetwarzanie pod eksperymenty.**  
- Obrazy zostały **przeskalowane do 32×32**, aby pasowały do ustawień
  „cifarowych” (VGG/ResNet).  
- Wejście: **1 kanał**, **10 klas**.  
- **Normalizacja per-kanał** (wyliczona na zbiorze uczącym); w praktyce
  często używa się mean $\approx 0.1307$, std $\approx 0.3081$ - takie wartości pojawiają
  się w przykładach referencyjnych PyTorcha [@pytorch].  
- **Podział train/val/test:** walidację wydzielono z treningu (5 000
  próbek) spójnie z innymi zbiorami.

**Dlaczego MNIST został użyty w pracy??**  
- Prosty, „czysty” zestaw do szybkich iteracji i testów **rotacji cyfr**
  (mało szumu, jednolity kontrast).  
- Umożliwia uczciwe porównanie **bazowych** (VGG/ResNet) z **wersjami
  cyklicznymi** (CyVGG/CyResNet) przy tym samym budżecie obliczeń.  
- W praktyce rotacje potrafią **mylić pary 6/9, 2/5** przy dużych
  kątach, jest to naturalny „edge case”, który dobrze obnaża różnice między
  *augmentacją*, a *architekturą*.

**Rotacje w eksperymentach.**  
Zastosowano kontrolowane scenariusze kątowe (szczegóły w rozdz. *Augmentacja
i protokół*):wariant **bez rotacji** (baseline), **małe/średnie obroty**
oraz **pełen zakres 0–360°**. Celem jest pokazanie, kiedy **architektura
cykliczna** daje przewagę nad samą augmentacją.


## GTSRB Gray (znaki drogowe w odcieniach szarości)

**German Traffic Sign Recognition Benchmark (GTSRB)** to zestaw znaków drogowych
z rzeczywistych nagrań, obejmujący **43 klasy**, z oficjalnym podziałem na część
uczącą i testową (IJCNN 2011) [@stallkamp2011gtsrb; @gtsrb_site]. W literaturze
często przytacza się także analizę „man vs. computer” z metrykami porównawczymi
[@stallkamp2012manvscomputer].

**Wariant „Gray” w tej pracy.**  
Na potrzeby eksperymentów wszystkie obrazy zostały **przeskalowane do 32×32**
i **skonwertowane do skali szarości** (1 kanał), tak aby pasowały do ustawień
„cifarowych” i umożliwiały **izolację wpływu rotacji** od informacji barwnej.
Zachowano **43 klasy**; walidację wydzielono z **oficjalnej** części treningowej
(spójnie z innymi zbiorami). Zastosowano **normalizację per-kanał** na zbiorze
uczącym.

**Dlaczego użyty został wariant Gray?**  
Kolor bywa silną wskazówką (np. czerwone obramowania, niebieskie tła), a celem
tej pracy jest **geometria** i sprawdzenie, co daje **architektura rotacyjnie
inwariantna** w porównaniu z bazową, bez „pomocy” informacji barwnej. Wersja Gray
ułatwia czyste porównania z **GTSRB RGB** (rozdz. poniżej), gdzie ewentualne różnice wynikają
właśnie z dostępności koloru.

**Wyzwania charakterystyczne dla GTSRB.**  
Nierównomierny rozkład klas, duża zmienność skali i oświetlenia,
perspektywa, rozmycie w ruchu, to wszystko to utrudnia proste uogólnianie i dobrze
testuje **stabilność względem rotacji** [@stallkamp2011gtsrb; @stallkamp2012manvscomputer].

**Rotacje w eksperymentach.**  
Wykorzystano scenariusze kątowe z rozdz. *Augmentacja i protokół* (bez rotacji,
małe/średnie obroty, połączenie róznych kombinacji kątów, pełen zakres 0–360°), aby porównać **VGG/ResNet** z
**CyVGG/CyResNet** w identycznym budżecie obliczeń.


## GTSRB RGB (znaki drogowe w kolorze)

**German Traffic Sign Recognition Benchmark (GTSRB)** w wersji kolorowej
to ten sam zestaw **43 klas** z oficjalnym podziałem train/test
[@stallkamp2011gtsrb; @gtsrb_site]. W pracy obrazy zostały
**przeskalowane do 32×32** (ustawienia „cifarowe”), z zachowaniem
**3 kanałów (RGB)**. Normalizacja wykonana **per-kanał** na zbiorze
uczącym; walidację wydzielono z części treningowej analogicznie jak dla
wariantu Gray [@stallkamp2012manvscomputer].

**Dlaczego została używa wersja RGB?**  
Kolor bywa silnym sygnałem (czerwone obramowania zakazów, żółte
trójkąty ostrzegawcze, niebieskie nakazy), więc użycie wariantu RGB pozwala
sprawdzić, na ile informacje barwne **kompensują** trudność związaną z
rotacjami - oraz jak bardzo **architektury rotacyjnie inwariantne**
(CyVGG/CyResNet) dalej poprawiają wyniki względem baz (VGG/ResNet).
Przyjęta procedura (ten sam rozmiar, te same podziały, ten sam
klasyfikator) pozwala porównywać **RGB vs Gray** 1:1.

**Wyzwania w praktyce.**  
Mimo przewagi koloru, duża zmienność **punktu widzenia**, **skali**,
**oświetlenia** i **rozmycia ruchu** pozostawiają problem rotacji jako
istotny czynnik trudności. Kolor pomaga odróżniać klasy o podobnym
kształcie, ale **nie zastępuje** inwariancji rotacyjnej.

**Rotacje w eksperymentach.**  
Wykorzystano te same scenariusze kątowe co wcześniej (baseline bez
rotacji, małe/średnie obroty, połączenie róznych kombinacji kątów, pełen zakres **0–360°**), gdyż celem jest
uczciwe porównanie **VGG/ResNet** i **CyVGG/CyResNet** przy identycznym
budżecie obliczeń.


## LEGO (obiekty 3D rzutowane na 2D)

Zbiór **Images of LEGO Bricks** (Kaggle) [@hazelzet_lego_kaggle] - obrazy
elementów LEGO renderowanych jako **rzuty 2D**,**skonwertowane do skali
szarości**. Na potrzeby pracy próbki zostały **przeskalowane do 96×96**,  
aby zachować detale kloców. Ustalono **50 klas** (1 kanał wejściowy), 
a walidacja wydzielona została z części treningowej analogicznie jak w innych zbiorach, 
zastosowana została również **normalizacja per-kanał**.

**Dlaczego LEGO?**  
- Obiekty mają **złożone kształty** i detale co sprawia, że jest to naturalny test
  „wrażliwości na orientację”.  
- W przeciwieństwie do MNIST (proste cyfry) i GTSRB (silny sygnał koloru),
  LEGO lepiej **izoluje geometrię** (kształt/układ wypustek, światłocień).  
- Dobrze pokazuje różnicę między podejściem **augmentacyjnym** a
  **architektonicznym** (CyVGG/CyResNet).

**Rotacje w eksperymentach.**  
Stosowano kontrolowane scenariusze kątowe opisane w rozdz. *Augmentacja i
protokół* (m.in. brak rotacji, małe/średnie obroty, połączenie róznych kombinacji kątów, 
pełen zakres 0–360°), co pozwala porównać bazowe modele (**VGG/ResNet**) z wersjami cyklicznymi
(**CyVGG/CyResNet**) przy tej samej części klasyfikacyjnej i budżecie
obliczeń.

**Uwaga praktyczna.**  
Przy **log-polarnych** przekształceniach i małej rozdzielczości blisko
środka pojawia się większa gęstość próbkowania, w preprocessing’u
zastosowana została interpolacja biliniarna i stały środek układu, aby ograniczyć
artefakty i zachować porównywalność między wariantami.


## Sposób augmentacji danych: zakresy rotacji, łączenie zbiorów

\newpage

# Architektury modeli
(VGG-E, ResNet-56, CyCNN)

**Przegląd.** W pracy wykorzystano bazowe architektury **VGG** (wariant **E**) i
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
i dopełnienie kanałów przez `F.pad(...)`. Zakończenie: **GAP** → warstwa
liniowa `64→C`.

W **CyResNet** wszystkie `nn.Conv2d` zastąpiono **`CyConv2d`** (w tym `conv1`
i konwolucje w `BasicBlock`). Pozostałe elementy (BN, ReLU, shortcut, GAP,
`Linear`) pozostajone zostały bez zmian względem wersji bazowej.


## Wersje cykliczne: **CyVGG-E** i **CyResNet-56**

- **Conv → CyConv.** Każdą `Conv2d` zastąpiono **`CyConv2d`**. Interfejs
  (rozmiary jąder, `stride`, `padding`) jest drop-in zgodny z `Conv2d`,
  więc topologia i klasyfikator są identyczne jak w bazach.

- **Implementacja warstwy.** `CyConv2d` opakowuje własną funkcję autograd
  (`CyConv2dFunction`) i wywołuje rozszerzenie CUDA
  `CyConv2d_cuda.forward/backward(...)`. Moduł posiada duży bufor roboczy na
  GPU (opisany w kodzie jako „Workspace for Cy-Winograd algorithm”). Wagi
  inicjalizowane są przez `xavier_uniform_`.

- **Uwaga dot. osi orientacji.** W kodzie modeli **nie ma jawnej dodatkowej
  osi „orientacja”** ani osobnego „poolingu po orientacjach”. Z poziomu
  PyTorch interfejs filtrów ma kształt `[C_out, C_in, k, k]` (jak w
  standardowym `Conv2d`). Mechanizmy rotacyjne - jeśli obecne - są
  enkapsulowane w jądrze CUDA `CyConv2d_cuda`, niewidocznym w plikach
  modeli.

- **Inwariancja w praktyce.** Po stronie modeli **GAP** oraz (opcjonalnie)
  dalsze uśrednianie w klasyfikatorze są identyczne jak w bazach - brak
  osobnego „poolingu po orientacjach” widocznego w kodzie modeli.

## Uzgodnienia I/O i selektor modeli

- **Kanały wejścia.** 1 kanał dla `mnist`, `mnist-custom`, `GTSRB-custom`, `LEGO`;
  3 kanały w pozostałych przypadkach (np. CIFAR, GTSRB RGB).
- **Liczba klas.** `MNIST`/`CIFAR-10`: 10; `GTSRB`: 43; `LEGO`: 50; `CIFAR-100`:
  100 - zgodnie z helperami `get_num_classes`.
- **Selektor modeli.** `get_model(model, dataset, classify=True)` zwraca jedną z
  wersji: `vgg*`, `cyvgg*`, `resnet*`, `cyresnet*`, zależnie od stringa modelu i
  nazwy zbioru danych.

## Standardowe CNN

Jako bazy zastosowano **VGG** (wariant **E / VGG-19**) i **ResNet-56**. Konwolucje
`3×3` z `padding=1`, w VGG z okresowym `MaxPool2d(2)`, w ResNecie zaś jest zmniejszanie
rozdzielczości przez `stride=2`. Po części splotowej **GAP** i **klasyfikator**
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

# Implementacja i środowisko eksperymentalne

### Szczegóły implementacyjne: warstwa `CyConv2d` (CUDA)

**Rozszerzenie - struktura**\
Warstwa korzysta z modułu C++/CUDA kompilowanego jako
`CyConv2d_cuda` z plików źródłowych `cycnn.cpp` i `cycnn_cuda.cu` (przy użyciu
`setuptools` i `BuildExtension`).

**Interfejs**\
W `cycnn.cpp` eksportowane są funkcje
`forward(...)` i `backward(...)` (pybind11). Przyjmują one `input`,
`weight`, `workspace` oraz `stride`, `padding`, `dilation`. Makra
sprawdzają, czy tensory są **CUDA** i **contiguous**, po czym wywoływane
są implementacje `cyconv2d_cuda_forward/backward`.

**Warstwa w PyTorch.**\
`CyConv2dFunction` (autograd) wywołuje
`CyConv2d_cuda.forward/backward` i przekazuje **workspace** oraz
parametry geometrii. Moduł `CyConv2d` przechowuje wagi o kształcie
`[C_out, C_in, k, k]` (inicjowane `xavier_uniform_`) i w `forward`
wykorzystuje `CyConv2dFunction.apply(...)`.

**Bufor roboczy.**\
`CyConv2d.workspace` jest to prealokowany tensor `float32`
na GPU o rozmiarze `1024*1024*1024` elementów (~ 4 GiB), opisany jako
„Workspace for Cy-Winograd algorithm”. Może to powodować **OOM** na
kartach graficznych z mniejszą ilością VRAM.

**Integracja z modelami.**\
Wszystkie `nn.Conv2d` w wariantach **CyVGG**
i **CyResNet** zostały zastąpione `CyConv2d` (m.in. `conv1` oraz
konwolucje znajdujące się w blokach). Topologia, BN/ReLU, GAP i klasyfikator są
zachowane 1:1 względem wersji bazowych.
W kodzie modeli nie ma **jawnej osi
„orientacja”** ani osobnego **poolingu po orientacjach**. Z poziomu
PyTorch wagi mają klasyczny kształt `[C_out, C_in, k, k]`. Jeśli
własności rotacyjne występują, są **enkapsulowane w jądrze CUDA**
(`cycnn_cuda.cu`) wywoływanym przez bindingi.

## Python, PyTorch

## Struktura projektu

## Automatyzacja: skrypty trenowania, testowania, ewaluacji

## Obsługa GPU, Docker, WSL

## Organizacja logów, modeli, confusion matrixów

\newpage
# Eksperymenty

## Scenariusze trenowania/testowania (opis JSON)

## Pomiar skuteczności (accuracy, macierze pomyłek)

## Śledzenie metryk: średnia, mediana, odchylenie standardowe

## Analiza skuteczności względem rotacji

## Ranking modeli

# Porównanie wyników

## CyCNN vs klasyczne CNN

## VGG vs CyVGG

## Resnet vs CyResNet

## CyVGG vs CyResNet

[...] cyresnet56 uczył się dłużej niż cyvgg19, ale dawał bardziej stabilne wyniki, w szczególności w przypadku funkcji aktywacji typu logpolar. 
I jak to się mówi - nie można zjeść ciastka i mieć go też (ang. „You can’t eat your
cake and have it too” [@kaczynski1995wp]).


## Wpływ transformacji (linearpolar vs logpolar)

## Wydajność na różnych zbiorach

# Wnioski

## Skuteczność rotacyjnych architektur

## Wnioski z automatyzacji i systematyzacji ewaluacji

## Propozycje dalszych badań

# Aneks

## Listingi kodów

## Dodatkowe wykresy, tablice wyników

\newpage
