param(
    [ValidateSet('caformer_s18','densenet121','hrnet_w18','all')][string]$Model='all',
    [switch]$CheckOnly,
    [switch]$Smoke
)
$ErrorActionPreference='Stop'
$runRoot=(Resolve-Path -LiteralPath $PSScriptRoot).Path
if ($runRoot -ne 'E:\比赛数据集\顶刊三模型对照_20261105') { throw 'Unexpected run root' }
$cacheRoot=Join-Path $runRoot 'temp'
foreach ($part in @('', 'cuda', 'huggingface', 'torch', 'xet')) {
    $cachePath=if ($part) { Join-Path $cacheRoot $part } else { $cacheRoot }
    New-Item -ItemType Directory -Path $cachePath -Force | Out-Null
}
$env:TEMP=$cacheRoot
$env:TMP=$cacheRoot
$env:TMPDIR=$cacheRoot
$env:XDG_CACHE_HOME=$cacheRoot
$env:CUDA_CACHE_PATH=Join-Path $cacheRoot 'cuda'
$env:HF_HOME=Join-Path $cacheRoot 'huggingface'
$env:HF_XET_CACHE=Join-Path $cacheRoot 'xet'
$env:TORCH_HOME=Join-Path $cacheRoot 'torch'
$env:HF_HUB_OFFLINE='1'
$env:HF_HUB_DISABLE_TELEMETRY='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$pythonExe='E:\比赛数据集\envs\cxr-smoke-cu128\Scripts\python.exe'
$scriptPath=Join-Path $runRoot 'run_journal_models.py'
$selection=if ($Model -eq 'all') { @('--run-all') } else { @('--model',$Model) }
if ($CheckOnly -or $Smoke) {
    $mode=if ($CheckOnly) { '--check-only' } else { '--smoke' }
    & $pythonExe -B $scriptPath @selection $mode
    if ($LASTEXITCODE -ne 0) { throw 'Preflight or smoke failed' }
    exit 0
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'pause_after_epoch.flag')) { throw 'Pause flag exists' }
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$outLog=Join-Path $runRoot ('stdout_'+$Model+'_'+$stamp+'.log')
$errLog=Join-Path $runRoot ('stderr_'+$Model+'_'+$stamp+'.log')
$arguments=@('-B',('"'+$scriptPath+'"'))+$selection
$worker=Start-Process -FilePath $pythonExe -ArgumentList $arguments -WindowStyle Hidden -WorkingDirectory $runRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$receipt=[pscustomobject]@{launcher_pid=$worker.Id;model=$Model;started=(Get-Date).ToString('o');stdout=$outLog;stderr=$errLog;python=$pythonExe;cache=$cacheRoot}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot ('launcher_'+$Model+'.json')) -Encoding utf8
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'launcher.json') -Encoding utf8
$receipt | ConvertTo-Json
