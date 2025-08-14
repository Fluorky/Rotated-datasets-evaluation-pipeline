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

## Cel i motywacja pracy

Powszechne konwolucyjne sieci neuronowe mają własności analizy obrazu z
zachowaniem niezmienniczości ze względu na translacje. Niestety brak w
tej chwili uznanych architektur pozwalających na rotacyjną
inwariantność. Celem pracy jest analiza efektywności nowych rozwiązań
proponowanych w literaturze ze względu na operacje rotacji (np.:
https://arxiv.org/pdf/2007.10588.pdf). Zadaniem dyplomanta będzie na
podstawie istniejącego zbioru uczącego (zdjęć, znaków drogowych lub
literek) zbudowanie nowego wzbogaconego o ich różnorakie rotacje.
Zaimplementowanie nowych architektur w pytorch lub tensorflow, a
następnie przeanalizowanie ich efektywności na wzbogaconym zbiorze
obrazów. Wyniki nowej architektury należy porównać do efektywności
standardowych sieci konwolucyjnych.

## Opis pracy

Praca magisterska wykorzystuje zaawansowane technologie i narzędzia
wspierające badania nad rotacyjnie inwariantnymi sieciami neuronowymi
oraz ich zastosowanie w przetwarzaniu obrazów. W realizacji projektu
zostaną zastosowane następujące technologie:

-   Język programowania Python -- podstawowe narzędzie do implementacji
    algorytmów i obsługi frameworków uczenia maszynowego dzięki jego
    wszechstronności i bogatemu ekosystemowi bibliotek, takich jak
    PyTorch i TensorFlow.

-   Frameworki uczenia maszynowego:

<!-- -->

-   PyTorch -- elastyczny framework umożliwiający szybkie prototypowanie
    oraz trenowanie sieci neuronowych.

-   TensorFlow -- kompleksowe narzędzie do budowy, trenowania i
    wdrażania modeli uczenia maszynowego.

-   Rotacyjnie inwariantne architektury sieci neuronowych -- analiza
    nowych rozwiązań badanych w literaturze naukowej, takich jak te
    opisane w publikacji „General E(2)-Equivariant Steerable CNNs".

<!-- -->

-   Wykorzystanie akceleracji GPU (np. NVIDIA) -- obliczenia zostaną
    przyspieszone dzięki zastosowaniu kart graficznych NVIDIA, które
    zapewniają wydajne środowisko dla intensywnych obliczeniowo operacji
    związanych z uczeniem głębokim. Frameworki takie jak PyTorch i
    TensorFlow wspierają CUDA oraz Tensor, co umożliwia efektywne wykorzystanie GPU.

-   Konteneryzacja za pomocą Docker -- narzędzie do tworzenia
    odizolowanych środowisk uruchomieniowych, które zapewni łatwość
    replikacji środowiska i współdzielenia projektu, również z obsługą
    GPU.


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
