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

    \IfFontExistsTF{Times New Roman}{
        \setmainfont{Times New Roman}
        \setsansfont{Times New Roman}
    }{
        \setmainfont{TeX Gyre Termes}
        \setsansfont{TeX Gyre Termes}
    }
    \IfFontExistsTF{Courier New}{
        \setmonofont[Scale=0.9]{Courier New}
    }{
        \setmonofont[Scale=0.9]{TeX Gyre Cursor}
    }
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
\newcommand{\magyardatum}{
    \number\year.\space
    \ifcase\month\or
    január\or február\or március\or április\or május\or június\or
    július\or augusztus\or szeptember\or október\or november\or december
    \fi
    \space\number\day.
}

\renewcommand{\maketitle}{
    \begin{titlepage}
        \centering
        \vspace*{-1.0cm}
        \includegraphics[width=6cm]{sze_logo.png}
        \vspace*{1.4cm}

        % --- CÍM ---
        {
            \fontsize{26pt}{40pt}\selectfont
            \bfseries
            \@title
        }

        \vspace{0.7cm}
        {\Large \bfseries TDK dolgozat}

        \vspace{1.2cm}

        {
            \fontsize{20pt}{26pt}\selectfont
            \bfseries
            \@author
        }

        \vfill

        \Large
        Konzulens: \\
        \@belsokonzulens

        \vfill

        \normalsize

        Győr \\
        \magyardatum
    \end{titlepage}
}

\makeatother
