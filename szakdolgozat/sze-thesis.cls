\ProvidesClass{sze-thesis}[2025 SZE Szakdolgozat Sablon (Készítette: Hunyadvári Ádám Radó)]
\LoadClass[
    12pt, 
    a4paper,
    oneside
]{scrreprt}

\RequirePackage{iftex}

\ifPDFTeX
    % PDFLaTeX specifikus csomagok
    \RequirePackage[utf8]{inputenc}
    \RequirePackage[T1]{fontenc}
    \RequirePackage{mathptmx} % Times New Roman helyettesítő
    \RequirePackage{courier}  % Courier helyettesítő
\else
    % XeLaTeX / LuaLaTeX specifikus csomagok
    \RequirePackage{fontspec}
    \ifLuaTeX
        \RequirePackage{luacode}
    \fi
    
    \setmainfont{Times New Roman}
    \setsansfont{Times New Roman}
    \setmonofont[Scale=0.9]{Courier New}
\fi

\RequirePackage[english]{babel}
\RequirePackage{graphicx}
\graphicspath{{images/}}
\RequirePackage{amsmath}

\RequirePackage[
    left=2.5cm,
    right=2.5cm,
    top=2.5cm,
    bottom=2.5cm
]{geometry}

\renewcommand*{\chapterformat}{\thechapter.\enskip}

\RequirePackage{csquotes}
\RequirePackage[backend=biber, style=iso-authoryear, sorting=nyt]{biblatex}

\RequirePackage{hyperref}

\hypersetup{
    colorlinks=true,
    linkcolor=black,
    urlcolor=black,
    citecolor=black,
}

\makeatletter
\AtBeginDocument{
    \hypersetup{
        pdftitle={\@title},
        pdfauthor={\@author}
    }
}

\newcommand{\belsokonzulens}[1]{\gdef\@belsokonzulens{#1}}
\newcommand{\kulsokonzulens}[1]{\gdef\@kulsokonzulens{#1}}
\newcommand{\tanszek}[1]{\gdef\@tanszek{#1}}

\renewcommand{\maketitle}{
    \begin{titlepage}
        \centering
        \includegraphics[width=6cm]{sze_logo.png}
        \vspace*{3cm}

        % --- CÍM ---
        {
            \fontsize{26pt}{40pt}\selectfont
            \bfseries
            \@title
        }

        \vspace{2cm}

        {
            \fontsize{20pt}{26pt}\selectfont
            \bfseries
            \@author
        }

        \vfill

        \Large
        Consultant: \\
        \@belsokonzulens

        \vfill

        \normalsize
        Széchenyi István Egyetem \\
        \@tanszek

        \vspace{1cm}

        Győr \\
        \today
    \end{titlepage}
}

\makeatother
