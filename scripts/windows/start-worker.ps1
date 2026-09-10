[CmdletBinding()]
param(
    [string]$EnvironmentName = "rag-textbook-qa",
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$ListenAddress
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

function Resolve-Executable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "Cannot find $Name. Install it or add it to PATH."
}

function Resolve-TailscaleAddress {
    param([Parameter(Mandatory = $true)][string]$TailscaleExecutable)

    $output = & $TailscaleExecutable ip -4 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Tailscale is not connected. Open Tailscale and try again."
    }

    foreach ($line in @($output)) {
        $candidate = $line.ToString().Trim()
        $parsedAddress = $null
        if (
            [System.Net.IPAddress]::TryParse($candidate, [ref]$parsedAddress) -and
            $parsedAddress.AddressFamily -eq
                [System.Net.Sockets.AddressFamily]::InterNetwork
        ) {
            return $candidate
        }
    }

    throw "Tailscale did not report an IPv4 address."
}

function Test-TcpPortInUse {
    param(
        [Parameter(Mandatory = $true)][string]$Address,
        [Parameter(Mandatory = $true)][int]$LocalPort
    )

    $targetAddress = [System.Net.IPAddress]::Parse($Address)
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    foreach ($listener in $listeners) {
        if ($listener.Port -ne $LocalPort) {
            continue
        }
        if (
            $listener.Address.Equals($targetAddress) -or
            $listener.Address.Equals([System.Net.IPAddress]::Any) -or
            $listener.Address.Equals([System.Net.IPAddress]::IPv6Any)
        ) {
            return $true
        }
    }
    return $false
}

function Assert-WorkerTokenConfigured {
    param([Parameter(Mandatory = $true)][string]$EnvPath)

    $tokenLine = Get-Content -LiteralPath $EnvPath | Where-Object {
        $_ -match '^\s*RAG_QA_WORKER_TOKEN\s*='
    } | Select-Object -Last 1

    if (-not $tokenLine) {
        throw "RAG_QA_WORKER_TOKEN is missing from project/.env."
    }

    $tokenValue = $tokenLine -replace '^\s*RAG_QA_WORKER_TOKEN\s*=', ''
    if (
        $tokenValue.Length -ge 2 -and
        (($tokenValue[0] -eq '"' -and $tokenValue[-1] -eq '"') -or
         ($tokenValue[0] -eq "'" -and $tokenValue[-1] -eq "'"))
    ) {
        $tokenValue = $tokenValue.Substring(1, $tokenValue.Length - 2)
    }

    if ([string]::IsNullOrEmpty($tokenValue)) {
        throw "RAG_QA_WORKER_TOKEN is empty in project/.env."
    }
    if ($tokenValue -ne $tokenValue.Trim()) {
        throw "RAG_QA_WORKER_TOKEN has leading or trailing whitespace."
    }
    foreach ($character in $tokenValue.ToCharArray()) {
        $codePoint = [int][char]$character
        if ($codePoint -lt 33 -or $codePoint -gt 126) {
            throw "RAG_QA_WORKER_TOKEN must contain printable ASCII characters only."
        }
    }
}

$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\..")
)
$projectFile = Join-Path $repositoryRoot "pyproject.toml"
$envFile = Join-Path $repositoryRoot "project\.env"

if (-not (Test-Path -LiteralPath $projectFile -PathType Leaf)) {
    throw "Cannot locate the repository from $PSScriptRoot."
}
if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    throw "Missing project/.env in $repositoryRoot."
}

Assert-WorkerTokenConfigured -EnvPath $envFile

$condaCandidates = @(
    $env:CONDA_EXE,
    (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
    (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
    (Join-Path $env:USERPROFILE "miniforge3\Scripts\conda.exe"),
    (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
    (Join-Path $env:LOCALAPPDATA "anaconda3\Scripts\conda.exe"),
    (Join-Path $env:LOCALAPPDATA "miniforge3\Scripts\conda.exe")
)
$condaExecutable = Resolve-Executable -Name "conda.exe" -Candidates $condaCandidates

if (-not $ListenAddress) {
    $tailscaleCandidates = @(
        (Join-Path $env:ProgramFiles "Tailscale\tailscale.exe"),
        (Join-Path $env:LOCALAPPDATA "Tailscale\tailscale.exe")
    )
    $tailscaleExecutable = Resolve-Executable -Name "tailscale.exe" -Candidates $tailscaleCandidates
    $ListenAddress = Resolve-TailscaleAddress -TailscaleExecutable $tailscaleExecutable
}

$parsedListenAddress = $null
if (
    -not [System.Net.IPAddress]::TryParse($ListenAddress, [ref]$parsedListenAddress) -or
    $parsedListenAddress.AddressFamily -ne
        [System.Net.Sockets.AddressFamily]::InterNetwork
) {
    throw "ListenAddress must be an IPv4 address."
}

if (Test-TcpPortInUse -Address $ListenAddress -LocalPort $Port) {
    throw "TCP ${ListenAddress}:$Port is already in use. Stop the existing Worker or select another port."
}

# project/.env is the source of truth. Remove stale process overrides only for
# this script and its child process; persistent user settings are not changed.
@(
    "RAG_QA_WORKER_TOKEN",
    "RAG_QA_EMBEDDING_MODEL",
    "RAG_QA_RERANKER_MODEL",
    "RAG_QA_DEVICE"
) | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}

Write-Host "Repository: $repositoryRoot"
Write-Host "Conda environment: $EnvironmentName"
Write-Host "Listen URL: http://${ListenAddress}:$Port"
Write-Host "Device: $Device"
Write-Host "Worker token: configured in project/.env"
Write-Host "Press Ctrl+C to stop the Worker."

& $condaExecutable run --no-capture-output -n $EnvironmentName `
    rag-qa --workspace $repositoryRoot worker serve `
    --host $ListenAddress --port $Port --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "Worker exited with code $LASTEXITCODE."
}
