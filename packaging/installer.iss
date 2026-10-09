; Inno Setup script for OpenVat.  Invoked by build_windows.ps1 with /DAppVersion=x.y.z
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppName=OpenVat
AppVersion={#AppVersion}
AppPublisher=OpenVat contributors
DefaultDirName={autopf}\OpenVat
DefaultGroupName=OpenVat
UninstallDisplayIcon={app}\OpenVat.exe
OutputDir=..\dist
OutputBaseFilename=OpenVat-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=openvat.ico
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"
Name: "assocstl"; Description: "Open .stl files with OpenVat"; GroupDescription: "File associations:"; Flags: unchecked

[Files]
Source: "..\dist\OpenVat\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\OpenVat"; Filename: "{app}\OpenVat.exe"
Name: "{group}\Uninstall OpenVat"; Filename: "{uninstallexe}"
Name: "{autodesktop}\OpenVat"; Filename: "{app}\OpenVat.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.stl\OpenWithProgids"; ValueType: string; ValueName: "OpenVat.stl"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assocstl
Root: HKA; Subkey: "Software\Classes\OpenVat.stl"; ValueType: string; ValueName: ""; ValueData: "STL model"; Flags: uninsdeletekey; Tasks: assocstl
Root: HKA; Subkey: "Software\Classes\OpenVat.stl\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\OpenVat.exe"" ""%1"""; Tasks: assocstl

[Run]
Filename: "{app}\OpenVat.exe"; Description: "Launch OpenVat"; Flags: nowait postinstall skipifsilent
