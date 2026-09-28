param([string]$OutDir, [switch]$Mono)
# Windows PowerShell 5.1: WinRT OneCore voices. One synthesis per line, 0.7 s silence between, 16 kHz mono via ffmpeg.
$ErrorActionPreference = 'Stop'
[void][Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
[void][Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | select -First 1
function Await($op, [type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(); $task.Result }

$voices = @{ A = 'Microsoft David'; B = 'Microsoft Zira'; C = 'Microsoft Mark' }
$script = @(
  @('A', 'Good morning everyone, thanks for joining the planning meeting today.'),
  @('A', 'We have three items on the agenda and about thirty minutes.'),
  @('A', 'First, the release schedule for the next quarter.'),
  @('A', 'Second, the budget for the new test lab.'),
  @('A', 'And third, hiring for the support team.'),
  @('B', 'Before we start, can I add a short update on the customer survey?'),
  @('B', 'It will only take two minutes, and it affects the release plan.'),
  @('C', 'I would also like to raise the question of the server migration.'),
  @('C', 'The old cluster runs out of support at the end of the year.'),
  @('C', 'If we move it, the lab budget has to change as well.'),
  @('A', 'Fair enough. Let us take the survey first, then the migration.'),
  @('A', 'After that we go back to the schedule as planned.'),
  @('A', 'Please keep each update short so we finish on time.'),
  @('C', 'Understood. I will keep the migration to five minutes.'),
  @('A', 'Thank you. Go ahead with the survey results, please.'),
  @('B', 'Most customers asked for faster search and better offline support.'),
  @('A', 'That is useful. We should put offline support into the next release.'),
  @('A', 'Let us write that down and move on to the migration.')
)
New-Item -ItemType Directory -Force $OutDir | Out-Null
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$all = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices
$list = Join-Path $OutDir 'list.txt'
$silence = Join-Path $OutDir 'silence.wav'
& ffmpeg -loglevel error -y -f lavfi -i 'anullsrc=r=16000:cl=mono' -t 0.7 -c:a pcm_s16le $silence
$lines = @()
for ($i = 0; $i -lt $script.Count; $i++) {
  $who = if ($Mono) { 'A' } else { $script[$i][0] }
  $synth.Voice = $all | ? { $_.DisplayName -eq $voices[$who] } | select -First 1
  $stream = Await ($synth.SynthesizeTextToStreamAsync($script[$i][1])) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
  $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
  $size = [uint32]$stream.Size
  [void](Await ($reader.LoadAsync($size)) ([uint32]))
  $bytes = New-Object byte[] $size; $reader.ReadBytes($bytes)
  $raw = Join-Path $OutDir ("raw{0:D2}.wav" -f $i); [IO.File]::WriteAllBytes($raw, $bytes)
  $seg = Join-Path $OutDir ("seg{0:D2}.wav" -f $i)
  & ffmpeg -loglevel error -y -i $raw -ar 16000 -ac 1 -c:a pcm_s16le $seg
  $lines += "file '$seg'"; $lines += "file '$silence'"
}
[IO.File]::WriteAllLines($list, $lines)
$out = Join-Path $OutDir ($(if ($Mono) { 'mono.wav' } else { 'three-voice.wav' }))
& ffmpeg -loglevel error -y -f concat -safe 0 -i $list -c:a pcm_s16le $out
Get-ChildItem $OutDir -Filter 'raw*.wav' | Remove-Item; Get-ChildItem $OutDir -Filter 'seg*.wav' | Remove-Item
Write-Output $out
