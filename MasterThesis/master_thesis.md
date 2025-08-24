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
konwolucyjnej sieci neuronowej bywa problemem. Rdzeń trudności to brak
naturalnej inwariantności względem rotacji: standardowe CNN-y „z
definicji” lepiej radzą sobie z przesunięciami niż z obrotami
[@goodfellow2016deep; @dumoulin2016guide].

W ostatnich latach pojawiło się kilka dróg domknięcia tej luki. Jedna to
poszerzanie danych o zrotowane przykłady - poprawia odporność, ale
wydłuża trening i nie gwarantuje uogólnienia na wszystkie kąty. Druga to
architektury z wbudowaną geometrią: sieci grupowo równoważne (G-CNN,
E(2)-equivariant) [@cohen2016group; @kim2020cycnn], sieci cykliczne
(CyCNN; w szczególności **CyVGG** i **CyResNet**) operujące na wielu
orientacjach oraz przekształcenia do układów polarnych (linear-polar i
log-polar), które „prostują” rotacje do przesunięć. Cel jest wspólny: by
model rozpoznawał „to samo” niezależnie od orientacji, bez agresywnego
dublowania danych.

Niniejsza praca skupia się na praktycznej weryfikacji tych podejść.
Przygotowano zbiory obejmujące m.in. odręcznie napisane cyfry, znaki
drogowe (w kolorze i w odcieniach szarości) oraz syntetyczne obiekty 3D
rzutowane na 2D (klocki LEGO), a następnie rozszerzono je o kontrolowane
rotacje. Zaimplementowano i porównano wybrane architektury rotacyjnie
inwariantne i ich warianty bazowe w **PyTorchu** [@paszke2019pytorch],
mierząc wpływ transformacji (linear-polar vs. log-polar), wyboru
architektury i zakresu kątów na jakość predykcji. Obliczenia realizowano
na kartach: **NVIDIA GeForce RTX 3070 Ti 8 GB** oraz **RTX 3060 12 GB**,
co skróciło czas trenowania i umożliwiło szeroki przegląd eksperymentów;
środowisko uruchomieniowe ustandaryzowano z użyciem **Dockera** dla
powtarzalności.

Celem pracy jest nie tylko pokazanie, że „da się” uzyskać odporność na
rotacje, ale przede wszystkim wskazanie, **kiedy** i **jakim kosztem**
ją osiągamy oraz które techniki przynoszą największy zysk względem
klasycznych CNN-ów, jak wpływają na stabilność i szybkość uczenia, a
także które konfiguracje są najpraktyczniejsze w realnych
zastosowaniach (OCR, rozpoznawanie znaków, analiza obiektów technicznych).
W dalszej części pracy przedstawiono podstawy, dane i augmentację,
architektury, środowisko eksperymentalne, protokoły ewaluacji oraz
wyniki z analizą i wnioskami.


## Cel i motywacja pracy

Konwolucyjne sieci neuronowe (CNN) charakteryzują się zdolnością do
analizy obrazów z zachowaniem niezmienniczości względem translacji.
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
umożliwiło przyspieszenie trenowania. Otrzymane wyniki zostały
porównane z rezultatami klasycznych sieci konwolucyjnych w celu oceny
realnych korzyści z rozwiązań inwariantnych względem rotacji.

## Opis pracy

Praca magisterska wykorzystuje zaawansowane technologie i narzędzia
wspierające badania nad rotacyjnie inwariantnymi sieciami neuronowymi
oraz ich zastosowaniem w przetwarzaniu obrazów. W realizacji projektu
zastosowano następujące rozwiązania technologiczne:

- **Język programowania Python** - podstawowe narzędzie do implementacji
  algorytmów oraz obsługi frameworków uczenia maszynowego, dzięki
  wszechstronności i ekosystemowi bibliotek [@python-docs].

  **Frameworki uczenia maszynowego:**
  - **PyTorch** - elastyczny framework do szybkiego prototypowania i
    trenowania modeli [@pytorch-docs].
  - **TensorFlow** - narzędzie do budowy, trenowania i wdrażania modeli
    ML/DL [@tensorflow-docs].

- **Modele cykliczne (CyCNN).** W pracy zostało przyjęte podejście, w
  którym obraz został przemapowany do współrzędnych $(\rho,\varphi)$.
  Dzięki temu obrót $R_\alpha$ staje się przesunięciem o $\alpha$ po osi
  $\varphi$. Konwolucje zostały zastąpione warstwami cylindrycznymi
  (CyConv) z cyklicznym paddingiem po $\varphi$. Dla każdego filtra
  zostało przygotowanych $n$ orientacji; odpowiedzi zostały złożone z
  dodatkową osią „orientacja”. Obrót wejścia powoduje cykliczny shift
  po tej osi, a pooling po orientacjach daje inwariancję względem
  rotacji. Został ustawiony stały środek układu polarnych, stała siatka
  próbkowania oraz biliniarna interpolacja; padding po $\varphi$ został
  ustawiony na cykliczny. Implementacja została wykonana w **PyTorchu**
  (punkt odniesienia: CyCNN [@kim2020cycnn]).

- **Wykorzystanie akceleracji GPU (NVIDIA)** - obliczenia zostały
  znacząco przyspieszone dzięki użyciu kart **RTX 3070 Ti 8 GB** oraz
  **RTX 3060 12 GB**. Frameworki wspierają **CUDA** oraz **cuDNN**, co
  umożliwia efektywne wykorzystanie zasobów GPU
  [@cuda-docs; @cudnn-docs].

- **Konteneryzacja za pomocą Dockera** - odizolowane środowiska
  uruchomieniowe ułatwiły replikację i współdzielenie projektu, także
  z obsługą GPU [@docker-docs].

\newpage

## Zakres tematyczny

Niniejsza praca dotyczy odporności modeli klasyfikacji obrazów na rotacje
w płaszczyźnie. Skupiono się na porównaniu klasycznych architektur z ich
wersjami rotacyjnie inwariantnymi oraz na wpływie przekształceń polarnych
na jakość predykcji. Badania zostały przeprowadzone na obrazach 2D i
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

- **Augmentacja i protokół:** zostały zdefiniowane zakresy kątów,
  liczba powtórzeń i podział na zbiory train/val/test (uczący/
  walidacyjny/testowy), z możliwością powtórzeń trenowań.

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

- Repozytoria z kodem, skryptami i plikami konfiguracyjnymi (PyTorch).
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
**VGG/ResNet** oraz wersje cykliczne **CyVGG/CyResNet**, wraz z
transformacjami linear-polar / log-polar. Rozdział **Implementacja i
środowisko** zawiera szczegóły techniczne: **PyTorch**, **CUDA i cuDNN**
(RTX 3070 Ti 8 GB, RTX 3060 12 GB), **Docker**, strukturę projektu i
skrypty. W **Eksperymentach** zostały zdefiniowane scenariusze, metryki
i sposób ewaluacji. Dalej, w **Porównaniu wyników**, zostały zestawione
modele (VGG vs. CyVGG, ResNet vs. CyResNet, wpływ transformacji) i
omówiona stabilność/czas. Na końcu **Wnioski** zbierają najważniejsze
**obserwacje** i wskazują kierunki dalszych badań; **Aneks** zawiera kody
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
przesuwa się po lokalnych fragmentach danych. W ten sposób uczą się prostych
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
powoduje, że wynik klasyfikacji nie zależy od dokładnej pozycji obiektu
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
odwracania jądra), mimo że w API funkcja nazywana jest `conv`
[@dumoulin2016guide].  
Nie ma to jednak znaczenia dla procesu uczenia - sieć i tak dobierze
właściwe wagi.

#### Stride, padding, rozmiary

Parametry „geometrii” warstwy:

- **padding** $p$ - ile pikseli dopisujemy na brzegach;
- **stride** $s$ - co ile pikseli przesuwamy okno;
- **dylacja** $d$ - „rozciąga” jądro (przerwy między próbkami).

#### Uwaga o „same/valid/stride” a ekwiwariancji\

Dokładna ekwiwariancja translacyjna zachodzi dla splotu bez zmian
rozmiaru. W praktyce **padding „same”**, **stride > 1** i **pooling**
wprowadzają drobne odchylenia (aliasing na siatce próbkowania), co
obniża „idealność” ekwiwariancji — efekt jest znany i opisywany
w literaturze [@dumoulin2016guide; @azulay2019small].

**Rozmiar wyjścia** (dla jądra $k\times k$):
$$
H'=\Big\lfloor \frac{H+2p-d\,(k-1)-1}{s}\Big\rfloor+1,\qquad
W'=\Big\lfloor \frac{W+2p-d\,(k-1)-1}{s}\Big\rfloor+1.
$$

**Typowe ustawienia.**  
- *valid*: $p=0$ - mapy cech maleją;  
- *same* (dla $s=1$): $p=\lfloor k/2\rfloor$ - $H'=H$, $W'=W$;  
- *stride $>1$*: wbudowane **podpróbkowanie** (mniej obliczeń, mniejsza rozdzielczość);  
- *dylacja $>1$*: większe **efektywne** pole widzenia bez nowych parametrów
  (częste w detekcji/segmentacji) [@dumoulin2016guide].

### Receptywne pole

Receptywne pole to fragment obrazu, który „widzi” dana aktywacja.
Teoretycznie rośnie ono z głębokością; dla kerneli $k_\ell$ i kroków $s_\ell$:
$$
R_1 = k_1,\qquad
R_\ell = R_{\ell-1} + (k_\ell-1)\!\!\prod_{j<\ell}s_j.
$$

W praktyce **efektywne** receptywne pole ma największy wpływ w centrum
i słabnie ku brzegom (rozkład zbliżony do Gaussa) - dlatego w bazowych
VGG/ResNet zostały dobrane głębokości tak, by objąć cały obiekt przy
rozdzielczościach użytych w eksperymentach. W wariantach **CyCNN** po
mapowaniu do $(\rho,\varphi)$ i z **cyklicznym paddingiem** po $\varphi$
sieć „widzi” pełny zakres orientacji bez artefaktów na brzegach
[@luo2016erf; @kim2020cycnn].

#### Receptywne pole w układzie polarnym

Po mapowaniu $(x,y)\!\to\!(\rho,\varphi)$ receptywne pole staje się
„wąskim paskiem” wzdłuż $\rho$ i stabilnym po $\varphi$. Dzięki temu
obrót $R_\alpha$ jest równoważny **przesunięciu** o $\alpha$ po osi
$\varphi$ (warstwy `CyConv` z **cyklicznym paddingiem** po $\varphi$ nie
„tną” informacji na brzegach), co wzmacnia ekwiwariancję rotacyjną
[@kim2020cycnn].


### Nieliniowości i normalizacja

Blok konwolucyjny został utrzymany **identyczny we wszystkich wariantach**
(bazowych i cyklicznych), bo celem porównania jest wpływ **rotacji**, a
nie doboru aktywacji. Normalizacja została zastosowana standardowo, aby
stabilizować uczenie; w wariantach **CyCNN** statystyki były liczone
łącznie po osi **orientacji**, tak aby **nie faworyzować żadnego kąta**
[@ioffe2015batchnorm; @kim2020cycnn].
W miejscach, gdzie stosowana jest normalizacja, statystyki są liczone
**wspólnie po osi orientacji**, tak żeby nie faworyzować żadnego kąta 
[@ioffe2015batchnorm; @kim2020cycnn].

### Pooling i część klasyfikacyjna

W modelach **CyCNN** inwariancja względem rotacji została osiągnięta
przez **agregację po orientacjach** (pooling po osi kątów). Dalej, dla
wszystkich modeli, zastosowano **global average pooling (GAP)** oraz
pojedynczą warstwę liniową w klasyfikatorze. Część klasyfikacyjna została
celowo taka sama w bazowych i cyklicznych wariantach, by izolować wpływ
części „rotacyjnej” [@lin2014network; @kim2020cycnn].

#### Pooling po orientacjach - szczegóły praktyczne

Agregacja po osi *orientacja* (avg lub max) realizuje **inwariancję**
rotacyjną. Wpływ na wynik ma liczba orientacji **n**:
większe **n** = dokładniejsza rozdzielczość kątowa (mniejszy błąd
zaokrąglenia $2\pi/n$), ale wyższy koszt obliczeń i pamięci.
W eksperymentach utrzymano identyczny klasyfikator za poolingiem, żeby
izolować wpływ części „rotacyjnej” [@kim2020cycnn].

### Regularizacja i trening

Protokół trenowania został **zamrożony** między wariantami (te same:
liczba epok, rozmiar batcha, budżet obliczeń, ewentualne wczesne
zatrzymanie - zgodnie z planem eksperymentu). Augmentacje ograniczono do
tych **niezależnych od rotacji** w testach „czysto architektonicznych”.
Augmentacja rotacją była używana tylko w eksperymentach kontrolnych
- żeby pokazać różnicę między „augmentacją” a „architekturą”.

### Triki architektoniczne (co faktycznie zmieniano)

- **Bazy:** VGG-19 (bloki $3{\times}3$) i ResNet-56 (wariant CIFAR, bloki $3{\times}3$).  
- **Wersje cykliczne:** podmiana `Conv` → `CyConv`, dodanie osi
  **orientacja** i **cykliczny padding** po $\varphi$; pozostałe elementy
  (głębokość, liczba kanałów) zostały dobrane tak, by utrzymać
  *porównywalny budżet parametrów/FLOPs* względem baz.  
- **Bez innych trików:** nie wprowadzano zmian niezwiązanych z rotacją,
  żeby nie mieszać efektów [@kim2020cycnn].

### Ekwawariancja translacyjna (kontrast do rotacyjnej)

Dla przesunięcia $\mathcal T_t$ i splotu zachodzi:
$$
\mathcal T_t(X) * K \;=\; \mathcal T_t\!\big(X*K\big),
$$
co tłumaczy, dlaczego klasyczne CNN dobrze radzą sobie z translacją
[@dumoulin2016guide]. Brak analogicznego mechanizmu dla rotacji
motywuje przekształcenia polarne i/lub modele cykliczne dalej.

#### Ekwiwariancja rotacyjna w dyskretnej grupie $C_n$

Dla indeksu orientacji $k\in\{0,\dots,n-1\}$ i kąta $\theta_k=\tfrac{2\pi k}{n}$
ekwiwariancję można zapisać jako

$$
\Phi\!\big(\mathcal{R}_{\theta_k} X\big)[\cdot,\;m]
\;=\; \Phi(X)[\cdot,\;m\!+\!k \bmod n],
$$

czyli **obrót wejścia = cykliczne przesunięcie po osi orientacji**. Następnie
pooling po tej osi znosi zależność od kąta (inwariancja) [@kim2020cycnn].


## Inwariancja translacyjna i rotacyjna

Niech $\mathcal T_t$ oznacza przesunięcie o wektor $t$, $\mathcal R_\alpha$
- obrót o kąt $\alpha$, a $\Phi$ - mapę cech/model.

**Ekwiwariancja** (przewidywalna zmiana reprezentacji):
$$
\Phi(\mathcal T_t X)=\Pi_t\,\Phi(X),\qquad
\Phi(\mathcal R_\alpha X)=\Pi_\alpha\,\Phi(X),
$$
gdzie $\Pi$ to działanie grupy na cechach (dla translacji - przesunięcie
map, dla rotacji - np. **cykliczny shift** po osi „orientacja”)
[@goodfellow2016deep; @bronstein2021gdl].

**Inwariancja** (wynik nie zależy od transformacji):
$$
\Phi(\mathcal T_t X)=\Phi(X),\qquad
\Phi(\mathcal R_\alpha X)=\Phi(X).
$$

W praktyce:  
- **translacja:** uśrednianie/podpróbkowanie (GAP, stride);  
- **rotacja:** (a) augmentacja o obroty, (b) architektura ze śledzeniem
  orientacji (oś „kąt” + pooling po orientacjach), (c) mapowanie do
  współrzędnych polarnych, gdzie obrót staje się przesunięciem po osi
  $\varphi$ [@dumoulin2016guide; @reddy1996logpolar; @kim2020cycnn].

### Linear-polar vs. log-polar (skrót)

- **Linear-polar**: równy przyrost w pikselach po $\rho$ i $\varphi$.
  Stabilne kąty, prosta implementacja; dobra pod **inwariancję rotacyjną**.
- **Log-polar**: skala rośnie logarytmicznie po $\rho$ — przesunięcia
  w skali stają się przesunięciami po osi promienia. Dobre pod **rotacje i skale**,
  ale bliżej środka pojawia się większa gęstość próbkowania i wrażliwość
  na wybór środka [@reddy1996logpolar].
- **Praktyka:** interpolacja biliniarna + **cykliczny padding po $\varphi$**,
  stały środek; blisko $\rho{=}0$ warto wygładzić/wykluczyć kilka próbek,
  żeby uniknąć „osobliwości” środka [@kim2020cycnn].
- 
## Problemy z rotacyjną inwariancją w klasycznych CNN

- **Kierunkowość filtrów.** Pojedynczy kernel jest wrażliwy na jedną
  orientację - bez dodatkowych mechanizmów sieć „gubi” obroty.  
- **Augmentacja nie domyka całości.** Rotacje pomagają, ale wydłużają
  trening i zostawiają „dziury” między kątami (przy małym kroku i
  ograniczonym budżecie).  
- **Aliasing/interpolacja.** Obracanie dyskretnych obrazów wprowadza
  artefakty i szum [@azulay2019small].  
- **Krawędzie i padding.** „same/zero” łamie symetrię przy brzegach -
  odpowiedzi nie są idealnie ekwiwariantne.  
- **Brak osi orientacji.** Standardowe CNN nie przechowują informacji,
  „pod jakim kątem” aktywacja została wykryta - trudno to później
  zwinąć do rozpoznawania niezależnego od kąta.

## Przegląd literatury (E(2)-equivariant, CyCNN)

**CyCNN (podejście użyte w pracy).** Obraz został przemapowany do
$(\rho,\varphi)$ i przetwarzany warstwami cylindrycznymi (**CyConv**) z
**cyklicznym paddingiem** po osi $\varphi$. Dla każdego filtra użyto
$n$ orientacji (grupa $C_n$). Obrót wejścia z $C_n$ wywołuje **cykliczny
shift** po osi orientacji (**ekwiwariancja**), a **pooling po orientacjach**
daje **inwariancję**. W badaniach zostały użyte **CyVGG** i **CyResNet**
[@kim2020cycnn].

**E(2)-equivariant / steerable CNNs (kontekst).** Sploty grupowe i
steerowalne jądra umożliwiają ekwiwariancję względem translacji i rotacji
(nawet dla ciągłych kątów) w grupie $\mathrm{E}(2)$. Wymagają projektu
jąder zgodnie z reprezentacjami grupy i zwykle większego kosztu
obliczeń. Taktowane są tu jako tło teoretyczne
[@cohen2016group; @weiler2019general; @cohen2019homogeneous].


# Opis zbiorów danych

## MNIST (cyfry odręczne)

## GTSRB Gray  (znaki drogowe w odcienach szarości)

## GTSRB RGB (znaki drogowe)

## LEGO (obiekty 3d rzutowane na 2d)

## Sposób augmentacji danych: zakresy rotacji, łączenie zbiorów

## Architektury modeli - stan faktyczny na podstawie kodu i szczegóły (VGG-E, ResNet-56, CyCNN)

**Przegląd.** Użyte zostały bazy **VGG** (wariant **E**) i **ResNet**
(wariant **56**) w ustawieniu „cifarowym” (obrazy 32×32): konwolucje 3×3
z `padding=1`, w VGG okresowy `MaxPool2d(2)`, w ResNet zmniejszanie
rozdzielczości przez `stride=2`. Na końcu **global average pooling**
(GAP) i **klasyfikator** (w VGG: dwuwarstwowe połączenia `512→512→C` z
dropoutem). W bazowym VGG został użyty wariant **z BatchNorm**
(VGG\_bn) [@simonyan2014vgg; @he2016resnet].

---

### VGG - wariant E (VGG-19, „3×3 everywhere”)

Układ wg [@simonyan2014vgg], dostosowany do 32×32 (CIFAR):

- **Blok 1:** 64, 64 → MaxPool (32→16)  
- **Blok 2:** 128, 128 → MaxPool (16→8)  
- **Blok 3:** 256×4 konwolucje → MaxPool (8→4)  
- **Blok 4:** 512×4 konwolucje → MaxPool (4→2)  
- **Blok 5:** 512×4 konwolucje → MaxPool (2→1)

Po części konwolucyjnej: `AdaptiveAvgPool(1,1)` → **klasyfikator**
`512→512→C` (dropout). W bazie: VGG\_bn.

### ResNet-56 (CIFAR, 6n+2 z n=9)

Schemat jak w [@he2016resnet] dla CIFAR:

- start: `Conv 3×3`, dalej **3 grupy** po **n=9** bloków `BasicBlock`,  
- **Grupa 1 (16 kanałów):** stride 1, 9 bloków,  
- **Grupa 2 (32 kanały):** pierwszy blok stride 2 (zmniejszenie rozdz.),  
- **Grupa 3 (64 kanały):** pierwszy blok stride 2.

`BasicBlock`: `Conv3×3 → BN → ReLU → Conv3×3 → BN` + skrót.  
Downsample: **opcja A** (CIFAR) - podpróbkowanie i dopełnienie kanałów.  
Zakończenie: GAP → **warstwa liniowa** `64→C`.

---

### Wersje cykliczne: **CyVGG-E** i **CyResNet-56**

- **Conv → CyConv.** Każdą `Conv2d` zastąpiono **`CyConv2d`**
  (warstwa cylindryczna; interfejs jak `Conv2d`: 3×3, `padding=1`,
  obsługa stride/dylacji).  

- **Oś orientacji.** Odpowiedzi składają się z dodatkową osią
  *orientacja* (liczba orientacji **n** z konfiguracji). Obrót wejścia
  odpowiada cyklicznemu przesunięciu po tej osi (**ekwawariancja**).  

- **Cykliczny padding po $\varphi$.** „Zawijanie” po osi kątowej usuwa
  artefakty brzegowe po mapowaniu do $(\rho,\varphi)$ [@kim2020cycnn].  

- **Inwariancja.** Przed klasyfikatorem zastosowano **agregację po
  orientacjach** (średnia/maksimum), co uniezależnia wynik od kąta.  

- **Reszta bez zmian.** Układ bloków, liczby kanałów i **klasyfikator**
  (GAP/Linear lub GAP + połączenia `512→512→C`) zostały zachowane jak w
  bazach, aby porównanie było uczciwe (zbliżony budżet parametrów/FLOPs)
  [@kim2020cycnn].

---

### Uzgodnienia I/O i implementacja

- **Kanały wejścia.** 1 kanał dla `MNIST`, `GTSRB_gray`, `LEGO`
  (w kodzie `GTSRB-custom`); 3 kanały dla `GTSRB_RGB`.
- **Liczba klas.** `MNIST` 10, `GTSRB` 43 (gray i GRB), `LEGO` 50 (zgodnie z
  `get_num_classes`).
- **Implementacja.** `CyConv2d` korzysta z dedykowanego kernela CUDA
  (`CyConv2d_cuda`) i bufora roboczego na GPU (przyspieszenie obliczeń,
  m.in. wariant Winograda). Część „rotacyjna” jest enkapsulowana w
  `CyConv2d` - kod modeli pozostaje w tym samym API [@kim2020cycnn].

  
## Standardowe CNN

Jako bazy zostały użyte **VGG** (wariant **E / VGG-19**) i
**ResNet-56** w układzie „cifarowym” (obrazy 32×32 oraz 96x97). 
Konwolucje 3×3 z `padding=1`, w VGG okresowy `MaxPool2d(2)`, w ResNet zmniejszanie
rozdzielczości przez `stride=2`. Po części konwolucyjnej zastosowano
**global average pooling (GAP)** i **klasyfikator** (w VGG: dwie
warstwy w pełni połączone `512→512→C` z dropoutem). W wariancie VGG
został użyty model **z BatchNorm** (VGG\_bn).  
Modele te są z natury **ekwawariantne translacyjnie** (dobrze znoszą
przesunięcia), ale nie posiadają wbudowanej inwariancji względem
rotacji - to jest punkt odniesienia dla wersji cyklicznych
[@simonyan2014vgg; @he2016resnet].

## Rotacyjnie inwariantne sieci (CyResNet, CyVGG)

Wersje cykliczne (**CyVGG-E**, **CyResNet-56**) powstały przez
**podmianę każdej `Conv2d` na `CyConv2d`** (warstwa cylindryczna).
Przetwarzanie odbywa się w układzie **$(\rho,\varphi)$**; po osi
**$\varphi$** został użyty **padding cykliczny**, aby „zawinąć” kąty.
Dla każdej warstwy została dodana **oś orientacji** (liczba orientacji
$n$ ustawiana w konfiguracji). Obrót wejścia o kąt z dyskretnego
zbioru $C_n$ przekłada się na **cykliczne przesunięcie** po tej osi
(**ekwawariancja**), a **agregacja po orientacjach** (avg/max) przed
klasyfikatorem daje **inwariancję** względem rotacji.  
Topologia bloków, liczby kanałów i część klasyfikacyjna zostały
utrzymane jak w bazach, żeby porównanie było **uczciwe** (zbliżony
budżet parametrów/FLOPs). Implementacja wykorzystuje dedykowany
kernel CUDA dla `CyConv2d`, co skraca czas obliczeń. Punktem
odniesienia jest koncepcja **CyCNN** (mapowanie polarne + warstwy
cylindryczne) [@kim2020cycnn].

## Transformacje polarne: linearpolar vs logpolar

# Implementacja i środowisko eksperymentalne

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
