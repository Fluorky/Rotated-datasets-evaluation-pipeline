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


\includegraphics[width=0.7 \textwidth]{media/image.png}

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

Obrazy otaczają nas z każdej strony: od zdjęć ze smartfonów, przez
monitoring miejski, zdjęcia produktów w bazach danych, systemy rozpoznawania jakości 
produktów na liniach produkcyjnych, po systemy wspomagania jazdy. Choć współczesne
modele rozpoznawania obrazu radzą sobie znakomicie w wielu zadaniach, w
praktyce często okazują się wrażliwe na pozornie drobne zmiany - jak
obrócenie obiektu o kilkanaście stopni czy nieznaczny przechył kamery.
To, co dla człowieka jest natychmiast rozpoznawalne (znak drogowy
widziany pod kątem, cyfra obrócona na kartce), dla klasycznej
konwolucyjnej sieci neuronowej bywa wyzwaniem. Kluczowy problem
sprowadza się do braku naturalnej inwariantności względem rotacji:
standardowe CNN-y „z definicji” lepiej radzą sobie z przesunięciami w
obrazie niż z obrotami.

W ostatnich latach zaproponowano szereg podejść, które mają tę lukę
wypełnić. Z jednej strony stosuje się rozszerzanie danych o zrotowane
przykłady, co poprawia odporność modeli kosztem dłuższego treningu i nie
zawsze gwarantuje uogólnienie na wszystkie kąty. Z drugiej strony
rozwijane są architektury, które wbudowują własności geometryczne w samą
sieć: od rozwiązań grupowo równoważnych (G-CNN, E(2)-equivariant),
przez sieci cykliczne operujące na wielu orientacjach, po
przekształcenia do układów polarnych (linear- oraz log-polar), które
„prostują” rotacje do przesunięć. Wspólnym mianownikiem jest dążenie do
tego, by model rozpoznawał „to samo” niezależnie od orientacji obiektu -
bez nadmiernego dublowania przykładów w zbiorze uczącym.

Niniejsza praca skupia się na praktycznej weryfikacji skuteczności
takich podejść. Przygotowano zbiory danych obejmujące m.in. odręczne
litery/cyfry, znaki drogowe (w kolorze i w odcieniach szarości) oraz
syntetyczne obiekty 3D rzutowane na 2D (np. klocki LEGO), a następnie
rozszerzono je o kontrolowane rotacje. Zaimplementowano i porównano
wybrane architektury rotacyjnie inwariantne oraz ich warianty bazowe w
**PyTorchu**, mierząc wpływ transformacji (linear-polar vs log-polar), wyboru
architektury modelu i zakresu kątów na jakość predykcji. Obliczenia realizowano
na kartach graficznych **NVIDIA GeForce RTX 3060 12 GB**, co pozwoliło skrócić czas
trenowania i przeprowadzić szeroki przegląd eksperymentów; środowisko
uruchomieniowe ustandaryzowano w oparciu o **Dockera**, aby zapewnić
powtarzalność wyników.

Celem pracy jest więc nie tylko pokazanie, że "da się" uzyskać
odporność na rotacje, ale przede wszystkim wskazanie, **kiedy** i
**jakim kosztem** ją osiągamy: które techniki dają największy zysk
względem klasycznych CNN-ów, jak wpływa to na stabilność i szybkość
uczenia oraz które konfiguracje są najpraktyczniejsze w realnych
zastosowaniach (OCR, rozpoznawanie znaków, analiza obiektów
technicznych). W kolejnych rozdziałach przedstawiono podstawy
teoretyczne, opis danych i augmentacji, architektury modeli, środowisko
eksperymentalne, protokoły ewaluacji oraz wyniki wraz z analizą i
wnioskami.

## Cel i motywacja pracy

Konwolucyjne sieci neuronowe (CNN) charakteryzują się zdolnością do analizy obrazów z zachowaniem 
niezmienniczości względem translacji. Jednak wciąż brakuje powszechnie uznanych architektur, które zapewniałyby 
inwariantność względem rotacji.
Celem niniejszej pracy jest analiza skuteczności nowych rozwiązań zaproponowanych w literaturze, 
ukierunkowanych na zapewnienie odporności modeli na rotację danych wejściowych (np. https://arxiv.org/pdf/2007.10588.pdf). 
W ramach pracy zostały przygotowane wzbogacone zbiory danych, obejmujące m.in. zdjęcia odręcznie napisanych liter, 
znaków drogowych (zarówno w kolorze, jak i w odcieniach szarości) oraz obiektów 3D rzutowanych na przestrzeń 2D (klocki LEGO), 
rozszerzone o różnorodne rotacje obrazów.

Na podstawie tych zbiorów danych przeprowadzono implementację i ewaluacja wybranych architektur rotacyjnie inwariantnych
z wykorzystaniem biblioteki PyTorch. Obliczenia zostały przeprowadzone przy użyciu kart graficznych NVIDIA GeForce RTX 3060 12GB, 
umożliwiających przyspieszenie procesów trenowania modeli. Otrzymane wyniki zostały porównane z rezultatami klasycznych 
sieci konwolucyjnych, w celu oceny realnych korzyści wynikających z zastosowania rozwiązań inwariantnych względem rotacji.

## Opis pracy

Praca magisterska wykorzystuje zaawansowane technologie i narzędzia wspierające badania nad rotacyjnie inwariantnymi sieciami 
neuronowymi oraz ich zastosowaniem w przetwarzaniu obrazów. 
W realizacji projektu zastosowano następujące rozwiązania technologiczne:

-   Język programowania Python – podstawowe narzędzie do implementacji algorytmów
    oraz obsługi frameworków uczenia maszynowego, dzięki swojej wszechstronności 
    i bogatemu ekosystemowi bibliotek, takich jak PyTorch i TensorFlow.

-   Frameworki uczenia maszynowego:

<!-- -->

-   PyTorch -- elastyczny framework umożliwiający szybkie prototypowanie
    oraz trenowanie sieci neuronowych.

-   TensorFlow -- kompleksowe narzędzie do budowy, trenowania i
    wdrażania modeli uczenia maszynowego.

-   Rotacyjnie inwariantne architektury sieci neuronowych -- analiza
    nowych rozwiązań badanych w literaturze naukowej, takich jak ta
    opisana w publikacji „General E(2)-Equivariant Steerable CNNs".

<!-- -->

-   Wykorzystanie akceleracji GPU (NVIDIA) - obliczenia zostały znacząco przyspieszone dzięki użyciu kart graficznych 
    NVIDIA GeForce RTX 3060 12GB, które zapewniają wydajne środowisko dla operacji obliczeniowo intensywnych. 
    Frameworki takie jak PyTorch i TensorFlow wspierają technologie CUDA oraz Tensor, umożliwiając efektywne 
    wykorzystanie zasobów GPU.

-   Konteneryzacja za pomocą Dockera – narzędzie do tworzenia odizolowanych środowisk 
    uruchomieniowych, które zapewniło łatwość replikacji środowiska oraz współdzielenia projektu, 
    także w konfiguracjach z obsługą GPU.


\newpage

## Zakres tematyczny

## Organizacja pracy

# Podstawy teoretyczne

## Wprowadzenie do sieci konwolucyjnych (CNN)

## Inwariancja translacyjna i rotacyjna

## Problemy z rotacyjną inwariancją w klasycznych CNN

## Przegląd literatury (np. E(2)-Equivariant CNNs, CyCNN)

# Opis zbiorów danych

## MNIST (cyfry odręczne)

## GTSRB Gray  (znaki drogowe w odcienach szarości)

## GTSRB RGB (znaki drogowe)

## LEGO (obiekty 3d rzutowane na 2s)

## Sposób augmentacji danych: zakresy rotacji, łączenie zbiorów

# Architektury modeli

## Standardowe CNN

## Rotacyjnie inwariantne sieci (np. CyResNet, CyVGG, G-CNN)

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
I jak to się mówi – nie można zjeść ciastka i mieć go też[21].

[21]: T.J. Kaczynski, *Industrial Society and Its Future*, 1995.

## Wpływ transformacji (linearpolar vs logpolar)

## Wydajność na różnych zbiorach

# Wnioski

## Skuteczność rotacyjnych architektur

## Wnioski z automatyzacji i systematyzacji ewaluacji

## Propozycje dalszych badań

# Bibliografia

# Aneks

## Listingi kodów

## Dodatkowe wykresy, tablice wyników
