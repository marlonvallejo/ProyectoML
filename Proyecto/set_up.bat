@echo off
REM Simple setup script to create a virtual environment and install requirements on Windows
echo.
echo === Proyecto - Crear entorno virtual e instalar dependencias ===

if exist .venv (
    echo Ya existe el directorio .venv, se usará el entorno existente.
) else (
    echo Creando entorno virtual .venv...
    python -m venv .venv
)

echo Activando entorno virtual...
call .venv\Scripts\activate

echo Actualizando pip e instalando dependencias desde requirements.txt
python -m pip install --upgrade pip
if exist requirements.txt (
    pip install --no-cache-dir -r requirements.txt
) else (
    echo No se encontró requirements.txt en el directorio actual.
)

echo.
echo Entorno configurado. Para activar el entorno manualmente use:
echo     call .venv\Scripts\activate
echo Para desactivarlo:
echo     deactivate

