---
header-includes:
  - \usepackage{graphicx}
---

\begin{titlepage}
\centering

\includegraphics[width=0.5\textwidth]{media/image.png}

\vspace{1cm}

{\LARGE Maciej Bujalski} \\[0.5cm]

{\large
Kierunek: informatyka \\
Specjalność: informatyka stosowana \\
Specjalizacja: aplikacje mobilne \\
Numer albumu: 386012
} \\[1cm]

\vspace{1cm}

{\Huge \textbf{Rotacyjnie inwariantne sieci neuronowe}} \\[1cm]

{\large
Praca magisterska wykonana pod kierunkiem \\
dr Krzysztofa Podlaskiego \\
w Katedrze Systemów Inteligentnych, WFiIS UŁ
} \\[2cm]

{\Large Łódź 2025}

\end{titlepage}

\newpage

\tableofcontents

\newpage

# Cel pracy

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

# Opis pracy

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
    TensorFlow wspierają CUDA, co umożliwia efektywne wykorzystanie GPU.

-   Konteneryzacja za pomocą Docker -- narzędzie do tworzenia
    odizolowanych środowisk uruchomieniowych, które zapewni łatwość
    replikacji środowiska i współdzielenia projektu, również z obsługą
    GPU.
