# Configuración de SonarCloud - Guía de Instalación

## ✅ Archivos Creados

1. `.github/workflows/sonarcloud.yml` - Workflow de GitHub Actions
2. `sonar-project.properties` - Configuración del proyecto

## Paso Final: Agregar el Token en GitHub

### Instrucciones:

1. **Ir a tu repositorio en GitHub:**

   - Abre: https://github.com/marlonvallejo/ProyectoML

2. **Navegar a Settings:**

   - Click en "Settings" (esquina superior derecha)

3. **Agregar el Secret:**

   - En el menú lateral izquierdo, busca "Secrets and variables"
   - Click en "Actions"
   - Click en "New repository secret"

4. **Configurar el Secret:**
   - **Name:** `SONAR_TOKEN`
   - **Value:** `5132ad48e66764335b754ffcf858c680690a16a1`
   - Click "Add secret"

## Hacer Push de los Archivos

```bash
git add .github/workflows/sonarcloud.yml sonar-project.properties
git commit -m "ci: Add SonarCloud integration for code quality analysis"
git push origin main
```

## Verificación

Después del push, GitHub Actions ejecutará automáticamente el análisis de SonarCloud.

**Para ver el resultado:**

1. Ve a la pestaña "Actions" en tu repositorio
2. Verás el workflow "SonarCloud Analysis" ejecutándose
3. El análisis tomará ~2-5 minutos
4. Los resultados aparecerán en: https://sonarcloud.io/project/overview?id=marlonvallejo_ProyectoML

## Qué Analiza SonarCloud

- **Code Smells:** Problemas de mantenibilidad
- **Bugs:** Errores potenciales en el código
- **Security Vulnerabilities:** Vulnerabilidades de seguridad
- **Code Coverage:** Cobertura de tests (si añades tests)
- **Code Duplications:** Código duplicado
- **Technical Debt:** Estimación de tiempo para resolver issues

## Ejecución Automática

El análisis se ejecutará automáticamente en:

- Cada `git push` a la rama `main`
- Cada Pull Request hacia `main`

## Nota Importante

⚠️ **NO subas el token al repositorio directamente**
El token debe estar SOLO en GitHub Secrets, nunca en el código fuente.
Por eso este archivo README se puede eliminar después de configurar el secret.
