; Ugyanabból a dist\LiveCapture mappából, mint a hordozható zip.
; Telepítés: %LocalAppData%\LiveCapture, rendszergazda nélkül.

#define AppName "LiveCapture"
#ifndef RepoDist
  #define RepoDist "..\dist\LiveCapture"
#endif

[Setup]
AppId={{A7C3E1B2-4F58-4D1A-9C0E-6B2F8A91D4E7}
AppName={#AppName}
AppVersion=1.0
DefaultDirName={localappdata}\LiveCapture
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
OutputDir=..\dist
OutputBaseFilename=LiveCapture-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\LiveCapture.exe

[Files]
Source: "{#RepoDist}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\LiveCapture.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\LiveCapture.exe"

[Run]
Filename: "{app}\LiveCapture.exe"; Description: "LiveCapture indítása"; Flags: nowait postinstall skipifsilent

[Code]
function VCRedistInstalled: Boolean;
var
  Installed: Cardinal;
begin
  Result := False;
  if RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) then
    Result := Installed = 1;
end;

function InitializeSetup: Boolean;
begin
  Result := True;
  if not VCRedistInstalled then
    MsgBox('A Microsoft Visual C++ 2015-2022 x64 Redistributable nincs telepítve.' + #13#10 +
      'Ha a LiveCapture hianyzo DLL-lel all le, telepitsd a Microsoft csomagjat, majd inditsd ujra.',
      mbInformation, MB_OK);
end;
