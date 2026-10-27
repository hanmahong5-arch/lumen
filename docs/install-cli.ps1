# Install the Lumen CLI binary from GitHub Releases (Windows, no Rust required).
#
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/hanmahong5-arch/lumen/main/docs/install-cli.ps1 | iex
#   $env:LUMEN_VERSION = 'lumen-cli-v0.3.2'; .\docs\install-cli.ps1
#   $env:PREFIX = "$env:LOCALAPPDATA\lumen\bin"; .\docs\install-cli.ps1
#
# Do NOT `cargo install lumen-cli` — that crates.io name is an unrelated project.

$ErrorActionPreference = 'Stop'

$Repo = if ($env:LUMEN_REPO) { $env:LUMEN_REPO } else { 'hanmahong5-arch/lumen' }
$Prefix = if ($env:PREFIX) { $env:PREFIX } else { Join-Path $env:LOCALAPPDATA 'lumen\bin' }

function Resolve-LumenVersion {
    if ($env:LUMEN_VERSION) { return $env:LUMEN_VERSION }

    $api = "https://api.github.com/repos/$Repo/releases"
    try {
        $latest = Invoke-RestMethod -Uri "$api/latest" -Headers @{ 'User-Agent' = 'lumen-install-cli' }
        if ($latest.tag_name -like 'lumen-cli-v*') { return $latest.tag_name }
    } catch {
        # fall through
    }

    try {
        $list = Invoke-RestMethod -Uri "$api?per_page=30" -Headers @{ 'User-Agent' = 'lumen-install-cli' }
        $hit = $list | Where-Object { $_.tag_name -like 'lumen-cli-v*' } | Select-Object -First 1
        if ($hit) { return $hit.tag_name }
    } catch {
        # fall through
    }

    return 'lumen-cli-v0.3.0'
}

$Version = Resolve-LumenVersion
$Asset = 'lumen-x86_64-pc-windows-msvc.exe'
$Base = "https://github.com/$Repo/releases/download/$Version"
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("lumen-install-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Tmp | Out-Null

try {
    Write-Host "-> downloading $Asset ($Version)"
    $binPath = Join-Path $Tmp $Asset
    $sumsPath = Join-Path $Tmp 'SHA256SUMS'
    Invoke-WebRequest -Uri "$Base/$Asset" -OutFile $binPath -UseBasicParsing
    Invoke-WebRequest -Uri "$Base/SHA256SUMS" -OutFile $sumsPath -UseBasicParsing

    $expectedLine = Get-Content $sumsPath | Where-Object { $_ -match [regex]::Escape($Asset) } | Select-Object -First 1
    if (-not $expectedLine) {
        throw "SHA256SUMS has no entry for $Asset"
    }
    $expected = ($expectedLine -split '\s+')[0].ToLowerInvariant()
    $actual = (Get-FileHash -Path $binPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expected -ne $actual) {
        throw "checksum mismatch for $Asset (expected $expected, got $actual)"
    }
    Write-Host "${Asset}: OK"

    New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
    $dest = Join-Path $Prefix 'lumen.exe'
    Copy-Item -Force $binPath $dest

    Write-Host "installed $dest"
    & $dest --version

    Write-Host ""
    Write-Host "Next:  lumen demo          # offline sample lifecycle HTML (no Kova required)"
    Write-Host "       lumen --help"

    $onPath = Get-Command lumen -ErrorAction SilentlyContinue
    if (-not $onPath -or $onPath.Source -ne $dest) {
        Write-Host ""
        Write-Host "Add to PATH for this session:"
        Write-Host "  `$env:Path = `"$Prefix;`$env:Path`""
        Write-Host "Persist via System Properties → Environment Variables, or:"
        Write-Host "  [Environment]::SetEnvironmentVariable('Path', `"$Prefix;`" + [Environment]::GetEnvironmentVariable('Path','User'), 'User')"
        Write-Host "Do NOT cargo install lumen-cli — that crates.io name is unrelated."
    }
}
finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
