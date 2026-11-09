[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [string]$PackageName
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot 'dist'
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot $OutputDirectory
}

if ([string]::IsNullOrWhiteSpace($PackageName)) {
    $PackageName = 'rl4co-display-advisor-{0}' -f (Get-Date -Format 'yyyyMMdd-HHmmss')
}

$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$stageRoot = Join-Path $outputRoot $PackageName
$zipPath = Join-Path $outputRoot ($PackageName + '.zip')

if ((Test-Path -LiteralPath $stageRoot) -or (Test-Path -LiteralPath $zipPath)) {
    throw "Output already exists. Choose another PackageName: $PackageName"
}

New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null

function Copy-RelativeFile {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $source = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required package file is missing: $RelativePath"
    }

    $destination = Join-Path $stageRoot $RelativePath
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

function Copy-FilteredTree {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [string[]]$ExcludedDirectoryNames = @('__pycache__', '.pytest_cache'),
        [string[]]$ExcludedFilePatterns = @('*.pyc', '*.pyo', '*.log', '*.tmp', '*.bak')
    )

    $sourceRoot = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
        throw "Required package directory is missing: $RelativePath"
    }

    Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | ForEach-Object {
        $relativeChild = $_.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        $segments = $relativeChild -split '[\\/]'
        $skipDirectory = $false

        foreach ($directoryName in $ExcludedDirectoryNames) {
            if ($segments -contains $directoryName) {
                $skipDirectory = $true
                break
            }
        }

        $skipFile = $false
        foreach ($pattern in $ExcludedFilePatterns) {
            if ($_.Name -like $pattern) {
                $skipFile = $true
                break
            }
        }

        if (-not $skipDirectory -and -not $skipFile) {
            $destination = Join-Path (Join-Path $stageRoot $RelativePath) $relativeChild
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
        }
    }
}

$rootFiles = @(
    '.env.example',
    '.gitignore',
    '.pylintrc',
    'app.py',
    'app_auth.py',
    'app_compat.py',
    'app_files.py',
    'app_gpu.py',
    'app_llm.py',
    'app_ollama.py',
    'app_pages.py',
    'app_stats.py',
    'app_training.py',
    'auth_module.py',
    'logging_config.py',
    'model_database.py',
    'pytest.ini',
    'README.md',
    'requirements.txt',
    'requirements-dev.txt',
    'test_all_combinations.py',
    'validate_combinations.py'
)
$rootFiles | ForEach-Object { Copy-RelativeFile $_ }

$configFiles = @(
    'config/__init__.py',
    'config/config.example.py',
    'config/database_init_with_auth.sql',
    'config/migration_gpu_unique_constraint.sql'
)
$configFiles | ForEach-Object { Copy-RelativeFile $_ }

Copy-FilteredTree 'modules' -ExcludedDirectoryNames @('__pycache__', '.pytest_cache', 'templates') -ExcludedFilePatterns @('*.pyc', '*.pyo', '*.log', '*.tmp', '*.bak', '*.md', '*.txt')
Copy-FilteredTree 'templates'
Copy-FilteredTree 'static/css'
Copy-FilteredTree 'static/js'
Copy-FilteredTree 'static/images'
Copy-FilteredTree 'tests' -ExcludedDirectoryNames @('__pycache__', '.pytest_cache', '_visualization_output') -ExcludedFilePatterns @('*.pyc', '*.pyo', '*.log', '*.tmp', '*.bak', 'validation_results_*')
Copy-FilteredTree 'scripts'

$rl4coFiles = @(
    'rl4co-main/.gitignore',
    'rl4co-main/CITATION.cff',
    'rl4co-main/LICENSE',
    'rl4co-main/pyproject.toml',
    'rl4co-main/run.py'
)
$rl4coFiles | ForEach-Object { Copy-RelativeFile $_ }
Copy-FilteredTree 'rl4co-main/rl4co' -ExcludedFilePatterns @('*.pyc', '*.pyo', '*.log', '*.tmp', '*.bak', '*.md', '*.txt')
Copy-FilteredTree 'rl4co-main/configs'

$runtimeDirectories = @(
    'checkpoints',
    'datasets',
    'lightning_logs',
    'logs',
    'static/model_plots'
)
foreach ($relativeDirectory in $runtimeDirectories) {
    $directory = Join-Path $stageRoot $relativeDirectory
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $directory '.gitkeep') -Value '' -Encoding UTF8
}

Compress-Archive -LiteralPath $stageRoot -DestinationPath $zipPath -CompressionLevel Optimal

$fileCount = (Get-ChildItem -LiteralPath $stageRoot -Recurse -File).Count
$totalBytes = (Get-ChildItem -LiteralPath $stageRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum
$zip = Get-Item -LiteralPath $zipPath
$zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()

[pscustomobject]@{
    PackageDirectory = $stageRoot
    ZipPath = $zip.FullName
    SourceFiles = $fileCount
    SourceBytes = $totalBytes
    ZipBytes = $zip.Length
    ZipSHA256 = $zipHash
}
