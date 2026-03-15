// TODO: Update with new Juice URLs once available
export const WINDOWS_URL = '';
export const WINDOWS_DOCUMENTID = '';
export const LINUX_URL = '';
export const LINUX_DOCUMENTID = '';


import windowsLogo from "./img/window-logo.png";
import ubuntuLogo from "./img/ubuntu.png";

export const SMDownloadList = [
  {
    os: "Windows",
    title: "Windows",
    logoTitle: "Windows",
    logo: windowsLogo,
    url: "./docs/stackmanager/setup/install-stackman#windows-1011",
    documentID: WINDOWS_DOCUMENTID,
    documentType: "exe",
    description: ".exe installer for Windows 10/11",
    href: WINDOWS_URL,
  },
  {
    os: "Linux",
    title: "Ubuntu",
    logoTitle: "Ubuntu",
    logo: ubuntuLogo,
    url: "./docs/stackmanager/setup/install-stackman#ubuntu-200422042404-lts",
    documentID: LINUX_DOCUMENTID,
    documentType: "deb",
    description: ".deb for Ubuntu 20.04/22.04/24.04 LTS",
    href: LINUX_URL,
  },
];
