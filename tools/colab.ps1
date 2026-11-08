# Atajo a la CLI oficial de Colab (corre en WSL porque no soporta Windows).
#   .\tools\colab.ps1 usage
#   .\tools\colab.ps1 new -s train --gpu T4
param([Parameter(ValueFromRemainingArguments = $true)] $ColabArgs)
$q = ($ColabArgs | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join ' '
wsl -d Ubuntu-22.04 -- bash -c "export PATH=`"`$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin`"; colab $q"
exit $LASTEXITCODE
