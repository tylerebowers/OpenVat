; Inno Setup 7 script for OpenVat: dist\OpenVat-<version>-setup.exe, a 64-bit
; installer for 64-bit Windows 10 (1809) / 11.
;
;     ISCC.exe /DAppVersion=0.1.0 packaging\installer.iss
;
; build_windows.ps1 (on Windows) and build_windows_wine.sh (on Linux, through
; Wine) run it after PyInstaller.  BuildDir is PyInstaller's one-folder build.

#if Ver < EncodeVer(7, 0, 0)
  #error Inno Setup 7 or newer is needed: https://jrsoftware.org/isdl.php
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef BuildDir
  #define BuildDir "..\build\windows\dist\OpenVat"
#endif

[Setup]
; The AppId Inno Setup 6 used by default (the AppName), so this installer
; updates an OpenVat installed by an older one.
AppId=OpenVat
AppName=OpenVat
AppVersion={#AppVersion}
AppPublisher=OpenVat contributors
AppPublisherURL=https://github.com/tylerebowers/OpenVat
AppSupportURL=https://github.com/tylerebowers/OpenVat/issues
AppUpdatesURL=https://github.com/tylerebowers/OpenVat/releases
; A 64-bit installer (new in Inno Setup 7): ArchitecturesAllowed and
; ArchitecturesInstallIn64BitMode then default to x64compatible, so it installs
; into the 64-bit Program Files on x64 and on Arm64 (x64 emulation) Windows.
SetupArchitecture=x64
; Qt 6 needs Windows 10 1809 or newer.
MinVersion=10.0.17763
DefaultDirName={autopf}\OpenVat
DefaultGroupName=OpenVat
DisableProgramGroupPage=yes
; Per machine (asks for administrator rights) or, picked in a dialog, just
; for the current user (no administrator rights needed).
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\OpenVat.exe
ChangesAssociations=yes
OutputDir=..\dist
OutputBaseFilename=OpenVat-{#AppVersion}-setup
SetupIconFile=openvat.ico
Compression=lzma2/max
SolidCompression=yes
; Light or dark, following Windows' app mode.
WizardStyle=modern dynamic

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "assocstl"; Description: "{cm:AssocFileExtension,OpenVat,.stl}"; GroupDescription: "File associations:"; Flags: unchecked

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\OpenVat"; Filename: "{app}\OpenVat.exe"
Name: "{autodesktop}\OpenVat"; Filename: "{app}\OpenVat.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.stl\OpenWithProgids"; ValueType: string; ValueName: "OpenVat.stl"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assocstl
Root: HKA; Subkey: "Software\Classes\OpenVat.stl"; ValueType: string; ValueName: ""; ValueData: "STL model"; Flags: uninsdeletekey; Tasks: assocstl
Root: HKA; Subkey: "Software\Classes\OpenVat.stl\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\OpenVat.exe,0"; Tasks: assocstl
Root: HKA; Subkey: "Software\Classes\OpenVat.stl\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\OpenVat.exe"" ""%1"""; Tasks: assocstl

[Run]
Filename: "{app}\OpenVat.exe"; Description: "{cm:LaunchProgram,OpenVat}"; Flags: nowait postinstall skipifsilent
