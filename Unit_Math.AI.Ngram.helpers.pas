unit Unit_Math.AI.Ngram.helpers;

interface

uses
  System.SysUtils,
  System.Classes,
  System.Generics.Defaults,
  System.Generics.Collections, math;

function LoadStringfromFile(const fileName: string): string;

{
  ApplyTopP:
  - tokens: candidate tokens (same order as weights)
  - weights: non-negative weights (probability-like scores)
  - TopP: nucleus threshold in (0, 1). E.g., 0.9 for 90% mass.
  - outTokens/outWeights: filtered lists (caller owns; created only on success)
  Returns True if filtered lists were produced; False otherwise (no allocation).
}
function ApplyTopP(const tokens: TList<string>; const weights: TList<Double>;
  TopP: Double; out outTokens: TList<string>;
  out outWeights: TList<Double>): Boolean;

function ParseFloatAnyFormat(const S: string;
  const Default: Double = 0.0): Double;

function SampleCounts(inner: TDictionary<string, Integer>; invTemp: Double;
  TopK: Integer; TopP: Double): string;

function IsCorePunct(const s: string): Boolean;

procedure AdjustWeightsForHeuristics(const tokens: TList<string>;
  const weights: TList<Double>; const prevToken: string);

implementation



procedure AdjustWeightsForHeuristics(const tokens: TList<string>;
  const weights: TList<Double>; const prevToken: string);
var
  i: Integer;
  prevIsPunct: Boolean;
begin
  if (tokens = nil) or (weights = nil) or (tokens.Count <> weights.Count) then
    Exit;

  prevIsPunct := IsCorePunct(prevToken);
  for i := 0 to tokens.Count - 1 do
  begin
    // Mild repetition penalty: if candidate == previous token, reduce weight
    if tokens[i] = prevToken then
      weights[i] := weights[i] * 0.35;

    // Avoid punctuation bursts: if prev is punct and candidate is punct, downweight
    if prevIsPunct and IsCorePunct(tokens[i]) then
      weights[i] := weights[i] * 0.25;
  end;
end;

// Existing functions below remain unchanged...
function SampleCounts(inner: TDictionary<string, Integer>; invTemp: Double;
  TopK: Integer; TopP: Double): string;
var
  pairs: TList<TPair<string, Integer>>;
  tokens: TList<string>;
  weights: TList<Double>;
  kv: TPair<string, Integer>;
  i: Integer;
  weight, sumW, draw: Double;
begin
  if (inner = nil) or (inner.Count = 0) then
    Exit('');

  pairs := TList<TPair<string, Integer>>.Create;
  tokens := TList<string>.Create;
  weights := TList<Double>.Create;
  try
    for kv in inner do
      pairs.Add(kv);

    if TopK > 0 then
      pairs.Sort(TComparer<TPair<string, Integer>>.Construct(
        function(const L, R: TPair<string, Integer>): Integer
        begin
          Result := R.Value - L.Value;
        end));

    if (TopK > 0) and (TopK < pairs.Count) then
      pairs.Count := TopK;

    sumW := 0.0;
    for i := 0 to pairs.Count - 1 do
    begin
      weight := Power(Max(1, pairs[i].Value), invTemp);
      tokens.Add(pairs[i].Key);
      weights.Add(weight);
      sumW := sumW + weight;
    end;

    // Note: AdjustWeightsForHeuristics is applied by caller where prevToken is known

    // Optional TopP
    var tk: TList<string> := nil;
    var wt: TList<Double> := nil;
    if ApplyTopP(tokens, weights, TopP, tk, wt) then
    begin
      tokens.Free; weights.Free;
      tokens := tk; weights := wt;
      sumW := 0.0;
      for i := 0 to weights.Count - 1 do
        sumW := sumW + weights[i];
    end;

    if sumW <= 0 then
      Exit(tokens[Random(tokens.Count)]);

    draw := Random * sumW;
    sumW := 0.0;
    for i := 0 to weights.Count - 1 do
    begin
      sumW := sumW + weights[i];
      if draw <= sumW then
        Exit(tokens[i]);
    end;
    Result := tokens[tokens.Count - 1];
  finally
    pairs.Free; tokens.Free; weights.Free;
  end;
end;


function IsCorePunct(const s: string): Boolean;
begin
  // Keep this list tight; expand only if needed
  Result :=
    (s = '.') or (s = ',') or (s = '!') or (s = '?') or (s = ';') or (s = ':') or
    (s = ')') or (s = '(') or (s = '"') or (s = '''') or (s = '-') or (s = '…');
end;



function LoadStringfromFile(const fileName: string): string;
var
  sl: TStringList;
begin
  sl := TStringList.Create;
  try
    sl.LoadFromFile(fileName, TEncoding.UTF8);
    Result := sl.Text;
  finally
    sl.Free;
  end;
end;

function ParseFloatAnyFormat(const S: string;
const Default: Double = 0.0): Double;
var
  fsCurrent, fsDot, fsComma: TFormatSettings;
  tmp: string;
  dotPos, commaPos: Integer;
  Value: Double;
begin
  fsCurrent := TFormatSettings.Create; // current system locale
  fsDot := TFormatSettings.Create;
  fsDot.DecimalSeparator := '.';
  fsDot.ThousandSeparator := ',';

  fsComma := TFormatSettings.Create;
  fsComma.DecimalSeparator := ',';
  fsComma.ThousandSeparator := '.';

  tmp := Trim(S);

  // Remove spaces commonly used as thousand separators in some locales
  tmp := StringReplace(tmp, ' ', '', [rfReplaceAll]);

  // If both '.' and ',' are present, decide which is decimal by last occurrence,
  // and remove the other as thousand separators.
  dotPos := LastDelimiter('.', tmp);
  commaPos := LastDelimiter(',', tmp);
  if (dotPos > 0) and (commaPos > 0) then
  begin
    if dotPos > commaPos then
      tmp := StringReplace(tmp, ',', '', [rfReplaceAll])
      // treat comma as thousand
    else
      tmp := StringReplace(tmp, '.', '', [rfReplaceAll]);
    // treat dot as thousand
  end;

  // Try parse in multiple ways: current locale, dot-locale, comma-locale
  if TryStrToFloat(tmp, Value, fsCurrent) then
    Exit(Value);
  if TryStrToFloat(tmp, Value, fsDot) then
    Exit(Value);
  if TryStrToFloat(tmp, Value, fsComma) then
    Exit(Value);

  // As a last resort, normalize any remaining separator to current decimal
  tmp := StringReplace(tmp, '.', fsCurrent.DecimalSeparator, [rfReplaceAll]);
  tmp := StringReplace(tmp, ',', fsCurrent.DecimalSeparator, [rfReplaceAll]);
  Result := StrToFloatDef(tmp, Default, fsCurrent);
end;

// Sort indices of "weights" in descending order using insertion sort for wide Delphi compatibility.
procedure SortIndicesByWeightDesc(indices: TList<Integer>;
const weights: TList<Double>);
var
  i, j, keyIdx: Integer;
begin
  for i := 1 to indices.Count - 1 do
  begin
    keyIdx := indices[i];
    j := i - 1;
    while (j >= 0) and (weights[indices[j]] < weights[keyIdx]) do
    begin
      indices[j + 1] := indices[j];
      Dec(j);
    end;
    indices[j + 1] := keyIdx;
  end;
end;

function ApplyTopP(const tokens: TList<string>; const weights: TList<Double>;
TopP: Double; out outTokens: TList<string>;
out outWeights: TList<Double>): Boolean;
var
  idx: TList<Integer>;
  i: Integer;
  total, cum: Double;
begin
  Result := False;
  outTokens := nil;
  outWeights := nil;

  if (TopP <= 0.0) or (TopP >= 1.0) or (tokens = nil) or (weights = nil) or
    (tokens.Count = 0) or (weights.Count <> tokens.Count) then
    Exit;

  idx := TList<Integer>.Create;
  try
    // Build index vector and sort descending by weight
    for i := 0 to weights.Count - 1 do
      idx.Add(i);
    SortIndicesByWeightDesc(idx, weights);

    // Total mass
    total := 0.0;
    for i := 0 to weights.Count - 1 do
      total := total + weights[i];
    if total <= 0 then
      Exit;

    // Take smallest set whose cumulative mass >= TopP
    outTokens := TList<string>.Create;
    outWeights := TList<Double>.Create;
    cum := 0.0;

    for i := 0 to idx.Count - 1 do
    begin
      outTokens.Add(tokens[idx[i]]);
      outWeights.Add(weights[idx[i]]);
      cum := cum + weights[idx[i]];
      if (cum / total) >= TopP then
        Break;
    end;

    Result := outTokens.Count > 0;
  finally
    idx.Free;
    if not Result then
    begin
      if outTokens <> nil then
        outTokens.Free;
      if outWeights <> nil then
        outWeights.Free;
    end;
  end;
end;

end.

  end.
