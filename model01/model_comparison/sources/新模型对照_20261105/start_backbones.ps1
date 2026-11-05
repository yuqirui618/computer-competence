param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$runRoot = $PSScriptRoot
if (-not $runRoot.StartsWith('E:\比赛数据集\新模型对照_20261105', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected run root' }
$runTemp = Join-Path $runRoot 'temp'
foreach ($cachePart in @('', 'cuda', 'huggingface', 'torch')) {
    $cachePath = if ($cachePart) { Join-Path $runTemp $cachePart } else { $runTemp }
    New-Item -ItemType Directory -Path $cachePath -Force | Out-Null
}
$env:TEMP = $runTemp
$env:TMP = $runTemp
$env:TMPDIR = $runTemp
$env:CUDA_CACHE_PATH = Join-Path $runTemp 'cuda'
$env:HF_HOME = Join-Path $runTemp 'huggingface'
$env:TORCH_HOME = Join-Path $runTemp 'torch'
$env:XDG_CACHE_HOME = $runTemp
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$pythonExe = 'E:\比赛数据集\envs\cxr-smoke-cu128\Scripts\python.exe'
$scriptPath = Join-Path $runRoot 'run_backbones.py'
if ($CheckOnly) {
    & $pythonExe -B $scriptPath --run-all --check-only
    if ($LASTEXITCODE -ne 0) { throw 'CPU preflight failed' }
    exit 0
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'pause_after_epoch.flag')) { throw 'Pause flag exists; review before resuming' }
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outLog = Join-Path $runRoot ('stdout_' + $runStamp + '.log')
$errLog = Join-Path $runRoot ('stderr_' + $runStamp + '.log')
$worker = Start-Process -FilePath $pythonExe -ArgumentList @('-B', ('"' + $scriptPath + '"'), '--run-all') -WindowStyle Hidden -WorkingDirectory $runRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
[pscustomobject]@{launcher_pid=$worker.Id;started=(Get-Date).ToString('o');stdout=$outLog;stderr=$errLog;python=$pythonExe;cache=$runTemp} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'launcher.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $runRoot 'launcher.json')



