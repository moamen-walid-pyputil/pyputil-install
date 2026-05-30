🚀 PyPUtil Install

Minimal toolkit for managing Python environments, packages, compilers, headers, and pip.


---

✨ Features

Tool	What it does

auto_installer	Auto-installs missing imports
package_installer	Install, upgrade, remove packages
pip_installer	Repair or reinstall pip
stdlib_installer	Install stdlib modules from CPython
python_installer	Manage standalone Python versions
pyheaders_installer	Install Python C headers
compiler_installer	Install GCC, Clang, Zig, Emscripten, NDK



---

📦 Install

pip install pyputil_install

Optional Extras

pip install pyputil_install[aiohttp]
pip install pyputil_install[color]
pip install pyputil_install[packaging]
pip install pyputil_install[all]


---

⚡ Quick CLI Usage

Package Manager

package-installer install requests
package-installer upgrade --all
package-installer uninstall flask
package-installer list --outdated


---

Pip Repair

pip-installer
pip-installer --version 21.3.1
pip-installer --uninstall


---

Stdlib Installer

stdlib-installer install json
stdlib-installer list
stdlib-installer remove json


---

Python Manager

python-installer install 3.11.5
python-installer list
python-installer switch 3.11.5


---

Headers Installer

headers-installer --version 3.11.0


---

🧠 Quick API Usage

Auto Installer

from pyputil_install.auto_installer import auto_install_sync

auto_install_sync()

import requests


---

Package Installer

from pyputil_install.package_installer import PackageInstaller

installer = PackageInstaller("requests")

installer.install()
print(installer.get_version())


---

Pip Installer

from pyputil_install.pip_installer import repair

repair()


---

Stdlib Installer

from pyputil_install.stdlib_installer import install_stdlib

install_stdlib("json")


---

Python Installer

from pyputil_install.python_installer.installer import PythonInstaller

installer = PythonInstaller()

installer.install("3.11.5")
installer.set_default("3.11.5")


---

Compiler Installer

from pyputil_install.compiler_installer import activate

activate("gcc", "14.2.0-2")


---

Supported Python Versions

CPython 3.9+

PyPy (partial support)



---

Storage Paths

Component	Default Path

Toolchains	~/.local/share/toolforge/toolchains/
Python Versions	~/.python_standalone/
Stdlib Cache	~/.cache/stdlib-installer/
Package Cache	~/.cache/packman/



---

Main Environment Variables

Variable	Purpose

TOOLFORGE_HOME	Custom install root
TOOLFORGE_BIN_DIR	Symlink directory
TOOLFORGE_SKIP_CHECKSUM	Disable checksum validation
TOOLFORGE_NO_SYMLINKS	Disable symlink creation
PIP_BREAK_SYSTEM_PACKAGES	Allow externally-managed installs



---

Exit Codes

Code	Meaning

0	Success
1	General error
2	Invalid arguments
3	Download/network error
4	Permission/extraction error
5	Installation failed



---

🏗️ Architecture

URL Layer
    ↓
Installer Layer
    ↓
Toolchain Layer
    ↓
Discovery Layer


---

License

MIT