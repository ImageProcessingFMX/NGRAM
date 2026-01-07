unit Unit_Math.AI.Ngram;

interface

uses
  System.SysUtils,
  System.Classes,
  System.Generics.Collections,
  System.Generics.Defaults,
  System.Math,
  System.Character,
  System.NetEncoding,
  System.IOUtils,
  System.StrUtils;

type

  TNGramType = (ngChar, ngWord);

  TSmoothing = (smLaplace, smWittenBell, smWittenBellFast);

  TNGramModel = class
  private
    FOrder: Integer;
    FType: TNGramType;
    FSmoothing: TSmoothing;
    FPreserveCase: Boolean;

    // ContextKey -> (NextToken -> Count)
    FCounts: TDictionary<string, TDictionary<string, Integer>>;
    FTotals: TDictionary<string, Integer>;
    FTokenToIndex: TDictionary<string, Integer>;

    // Global unigram stats
    FGlobalCounts: TDictionary<string, Integer>;
    FGlobalTotal: Integer;

    // Token alphabet (unique tokens)
    FAlphabet: TList<string>;

    // Helpers
    function NormalizeText(const s: string): string;
    function Tokenize(const s: string): TArray<string>;
    function InsertSentenceBoundaries(const tokens: TArray<string>)
      : TArray<string>;
    function Detokenize(const tokens: TArray<string>): string;

    function ContextKey(const tokens: TArray<string>; Count: Integer): string;
    procedure AddContext(const ContextKey: string; const nextToken: string);

    function GetAlphabetIndex(const token: string): Integer;
    function TryGetInner(const ContextKey: string;
      out inner: TDictionary<string, Integer>): Boolean;

    // Utility
    function SumCounts(inner: TDictionary<string, Integer>): Integer;
    function EnsureTokenId(const token: string): Integer;

    // Persistence helpers
    function EncodeB64(const s: string): string;
    function DecodeB64(const s: string): string;

    // Sampling utils
    function ApplyTopP(const tokens: TList<string>;
      const weights: TList<Double>; TopP: Double; out outTokens: TList<string>;
      out outWeights: TList<Double>): Boolean;

    // Sampling
    function SampleNextToken(const fullContextTokens: TArray<string>;
      Temperature: Double; TopK: Integer; TopP: Double): string;
    function SampleLaplace(inner: TDictionary<string, Integer>; total: Integer;
      Temperature: Double; TopK: Integer; TopP: Double;
      const prevToken: string): string;
    function SampleWittenBell(ctxInner: TDictionary<string, Integer>;
      ctxTotal: Integer; backoffInner: TDictionary<string, Integer>;
      backoffTotal: Integer; Temperature: Double; TopK: Integer; TopP: Double;
      const prevToken: string): string;
    function SampleWittenBellFast(ctxInner: TDictionary<string, Integer>;
      ctxTotal: Integer; backoffInner: TDictionary<string, Integer>;
      backoffTotal: Integer; Temperature: Double; TopK: Integer; TopP: Double;
      const prevToken: string): string;

    function ContextKeySlice(const tokens: TArray<string>;
      startIdx, Count: Integer): string;

  public
    // Constructors
    constructor Create(order: Integer); overload;
    // default char-level, Laplace smoothing
    constructor Create(order: Integer; gramType: TNGramType;
      preserveCase: Boolean = False;
      smoothing: TSmoothing = smLaplace); overload;
    destructor Destroy; override;

    // Training and generation
    procedure Train(const text: string);
    procedure TrainFile(const FileName: string);
    procedure TrainFiles(const Files: TArray<string>);
    function Generate(const seed: string; GenLength: Integer;
      Temperature: Double = 1.0; TopK: Integer = 0; TopP: Double = 0.0): string;

    // Persistence
    procedure SaveToFile(const FileName: string);
    procedure LoadFromFile(const FileName: string);

    // Diagnostics
    procedure GetStats(out Contexts: Integer; out Alphabet: Integer;
      out GlobalTotal: Integer);

    // Others
    function GetStatsasString: string;
    procedure Clear;

    procedure ComputeSparsityReport(outLines: TStrings; MaxLines: Integer = 0);

    // Properties
    function AlphabetSize: Integer;
    property order: Integer read FOrder;
    property gramType: TNGramType read FType;
    property smoothing: TSmoothing read FSmoothing write FSmoothing;
    property preserveCase: Boolean read FPreserveCase write FPreserveCase;
  end;

implementation

uses Unit_Math.AI.Ngram.helpers;

const
  CONTEXT_DELIM = Char(31); // Unit Separator
  BOS_TOKEN = '<s>';
  EOS_TOKEN = '</s>';
  FILE_MAGIC = 'NGRAM.VERSION.1';

constructor TNGramModel.Create(order: Integer);
begin
  Create(order, ngChar, False, smLaplace);
end;

procedure TNGramModel.TrainFile(const FileName: string);
var
  text: string;
begin
  text := TFile.ReadAllText(FileName, TEncoding.UTF8);
  if text.Trim <> '' then
    Train(text);
end;

procedure TNGramModel.TrainFiles(const Files: TArray<string>);
var
  f: string;
begin
  for f in Files do
    TrainFile(f);
end;

procedure TNGramModel.Clear;
var
  key: string;
  inner: TDictionary<string, Integer>;
begin
  for key in FCounts.Keys do
  begin
    inner := FCounts.Items[key];
    inner.Free;
  end;
  FCounts.Clear;
  FTotals.Clear;
  FGlobalCounts.Clear;
  FGlobalTotal := 0;
  FAlphabet.Clear;
  FTokenToIndex.Clear;
end;

procedure TNGramModel.ComputeSparsityReport(outLines: TStrings;
  MaxLines: Integer = 0);
var
  ctxKey: string;
  inner: TDictionary<string, Integer>;
  totalContexts: Integer;
  sumTotals: Int64;
  sumBranching: Int64;
  singletons: Int64;
  zeroOrEmpty: Int64;
  maxBranching, v, ctxTotal: Integer;
  hasBOS, hasEOS: Boolean;
begin
  totalContexts := 0;
  sumTotals := 0;
  sumBranching := 0;
  singletons := 0;
  zeroOrEmpty := 0;
  maxBranching := 0;

  for ctxKey in FCounts.Keys do
  begin
    Inc(totalContexts);
    inner := FCounts.Items[ctxKey];
    ctxTotal := 0;
    if inner <> nil then
    begin
      for v in inner.Values do
        Inc(ctxTotal, v);
      sumTotals := sumTotals + ctxTotal;
      sumBranching := sumBranching + inner.Count;
      if inner.Count > maxBranching then
        maxBranching := inner.Count;
      if ctxTotal <= 1 then
        Inc(singletons);
    end
    else
    begin
      Inc(zeroOrEmpty);
    end;
  end;

  hasBOS := GetAlphabetIndex(BOS_TOKEN) <> -1;
  hasEOS := GetAlphabetIndex(EOS_TOKEN) <> -1;

  outLines.Add('Sparsity report:');
  outLines.Add('  Contexts: ' + totalContexts.ToString);
  outLines.Add('  Alphabet: ' + FAlphabet.Count.ToString);
  outLines.Add('  GlobalTotal: ' + FGlobalTotal.ToString);
  if totalContexts > 0 then
  begin
    outLines.Add('  Avg total per context: ' + FormatFloat('0.00',
      sumTotals / totalContexts));
    outLines.Add('  Avg branching factor: ' + FormatFloat('0.00',
      sumBranching / totalContexts));
    outLines.Add('  Max branching factor: ' + maxBranching.ToString);
    outLines.Add('  Singleton contexts (%): ' + FormatFloat('0.00',
      100.0 * singletons / totalContexts));
    if zeroOrEmpty > 0 then
      outLines.Add('  Empty contexts: ' + zeroOrEmpty.ToString);
  end;
  hasBOS := FTokenToIndex.ContainsKey(BOS_TOKEN);
  hasEOS := FTokenToIndex.ContainsKey(EOS_TOKEN);
  outLines.Add('  BOS present: ' + BoolToStr(hasBOS, True));
  outLines.Add('  EOS present: ' + BoolToStr(hasEOS, True));
end;

constructor TNGramModel.Create(order: Integer; gramType: TNGramType;
  preserveCase: Boolean; smoothing: TSmoothing);
begin
  inherited Create;
  if order < 1 then
    raise Exception.Create('Order must be >= 1');
  FOrder := order;
  FType := gramType;
  FPreserveCase := preserveCase;
  FSmoothing := smoothing;

  FCounts := TDictionary < string, TDictionary < string, Integer >>.Create;
  FTotals := TDictionary<string, Integer>.Create;
  FGlobalCounts := TDictionary<string, Integer>.Create;
  FTokenToIndex := TDictionary<string, Integer>.Create;

  FGlobalTotal := 0;
  FAlphabet := TList<string>.Create;
end;

destructor TNGramModel.Destroy;
var
  inner: TDictionary<string, Integer>;
  key: string;
begin
  for key in FCounts.Keys do
  begin
    inner := FCounts.Items[key];
    inner.Free;
  end;
  FCounts.Free;
  FTotals.Free;
  FGlobalCounts.Free;
  FAlphabet.Free;
  FTokenToIndex.Free;
  inherited Destroy;
end;

function TNGramModel.NormalizeText(const s: string): string;
var
  i: Integer;
  c: Char;
  sb: TStringBuilder;
begin
  sb := TStringBuilder.Create;
  try
    for i := 1 to Length(s) do
    begin
      c := s[i];
      // Preserve Unicode; normalize whitespace to space
      if TCharacter.IsWhiteSpace(c) then
      begin
        sb.Append(' ');
      end
      else
      begin
        if FPreserveCase then
          sb.Append(c)
        else
          sb.Append(TCharacter.ToLower(c));
      end;
    end;

    // Collapse multiple spaces (use StrUtils.ReplaceStr for broad compatibility)
    Result := sb.ToString;
    while Pos('  ', Result) > 0 do
      Result := ReplaceStr(Result, '  ', ' ');
    Result := Trim(Result);
  finally
    sb.Free;
  end;
end;

function TNGramModel.Tokenize(const s: string): TArray<string>;
var
  text: string;
  tokens: TList<string>;
  i: Integer;
  c: Char;
  current: TStringBuilder;
begin
  text := NormalizeText(s);

  // Character-level: each char is a token
  if FType = ngChar then
  begin
    SetLength(Result, Length(text));
    for i := 1 to Length(text) do
      Result[i - 1] := text[i];
    Exit;
  end;

  // Word-level: Unicode-aware
  tokens := TList<string>.Create;
  current := TStringBuilder.Create;
  try
    for i := 1 to Length(text) do
    begin
      c := text[i];

      if c = ' ' then
      begin
        if current.Length > 0 then
        begin
          tokens.Add(current.ToString);
          current.Clear;
        end;
        // skip adding space as a token
      end
      else if TCharacter.IsLetterOrDigit(c) or (c = '''') then
      begin
        current.Append(c);
      end
      else if TCharacter.IsPunctuation(c) then
      begin
        if current.Length > 0 then
        begin
          tokens.Add(current.ToString);
          current.Clear;
        end;
        tokens.Add(c);
      end
      else
      begin
        // Any other printable symbol becomes its own token
        if current.Length > 0 then
        begin
          tokens.Add(current.ToString);
          current.Clear;
        end;
        tokens.Add(c);
      end;
    end;

    if current.Length > 0 then
      tokens.Add(current.ToString);

    Result := tokens.ToArray;
  finally
    tokens.Free;
    current.Free;
  end;
end;

function TNGramModel.InsertSentenceBoundaries(const tokens: TArray<string>)
  : TArray<string>;
var
  outTokens: TList<string>;
  i: Integer;
  t: string;
  procedure StartSentence;
  begin
    outTokens.Add(BOS_TOKEN);
  end;
  procedure EndSentence;
  begin
    outTokens.Add(EOS_TOKEN);
  end;

begin
  if FType <> ngWord then
  begin
    // No sentence markers for character-level
    Result := tokens;
    Exit;
  end;

  outTokens := TList<string>.Create;
  try
    StartSentence;
    for i := 0 to Length(tokens) - 1 do
    begin
      t := tokens[i];
      outTokens.Add(t);
      if (t = '.') or (t = '!') or (t = '?') then
      begin
        EndSentence;
        StartSentence;
      end;
    end;
    // If we didn’t close the last sentence, end it
    if (outTokens.Count > 0) and (outTokens[outTokens.Count - 1] <> EOS_TOKEN)
    then
      EndSentence;

    Result := outTokens.ToArray;
  finally
    outTokens.Free;
  end;
end;

function TNGramModel.Detokenize(const tokens: TArray<string>): string;
var
  sb: TStringBuilder;
  i: Integer;
  t, prev: string;
  needsSpace: Boolean;
  function IsNoLeadingSpace(const s: string): Boolean;
  begin
    Result := (s = '.') or (s = ',') or (s = '!') or (s = '?') or (s = ';') or
      (s = ':') or (s = ')') or (s = ']') or (s = '}');
  end;
  function IsNoTrailingSpace(const s: string): Boolean;
  begin
    Result := (s = '(') or (s = '[') or (s = '{') or (s = '"');
  end;

begin
  if FType = ngChar then
  begin
    sb := TStringBuilder.Create;
    try
      for i := 0 to Length(tokens) - 1 do
        sb.Append(tokens[i]);
      Result := sb.ToString;
    finally
      sb.Free;
    end;
    Exit;
  end;

  sb := TStringBuilder.Create;
  try
    prev := '';
    for i := 0 to Length(tokens) - 1 do
    begin
      t := tokens[i];
      // Skip sentence markers in output
      if (t = BOS_TOKEN) or (t = EOS_TOKEN) then
        Continue;

      if sb.Length = 0 then
      begin
        sb.Append(t);
      end
      else
      begin
        needsSpace := True;
        if IsNoLeadingSpace(t) then
          needsSpace := False;
        if IsNoTrailingSpace(prev) then
          needsSpace := False;

        if needsSpace then
          sb.Append(' ');
        sb.Append(t);
      end;
      prev := t;
    end;
    Result := sb.ToString;
  finally
    sb.Free;
  end;
end;

function TNGramModel.ContextKey(const tokens: TArray<string>;
  Count: Integer): string;
var
  i, startIdx: Integer;
  sb: TStringBuilder;
begin
  if Count <= 0 then
    Exit('');

  startIdx := Length(tokens) - Count;
  if startIdx < 0 then
    startIdx := 0;

  sb := TStringBuilder.Create;
  try
    for i := startIdx to Length(tokens) - 1 do
    begin
      sb.Append(tokens[i]);
      if i < Length(tokens) - 1 then
        sb.Append(CONTEXT_DELIM);
    end;
    Result := sb.ToString;
  finally
    sb.Free;
  end;
end;

procedure TNGramModel.AddContext(const ContextKey: string;
  const nextToken: string);
var
  inner: TDictionary<string, Integer>;
  current: Integer;
begin
  if not FCounts.TryGetValue(ContextKey, inner) then
  begin
    inner := TDictionary<string, Integer>.Create;
    FCounts.Add(ContextKey, inner);
    FTotals.Add(ContextKey, 0);
  end;

  if inner.TryGetValue(nextToken, current) then
    inner.AddOrSetValue(nextToken, current + 1)
  else
    inner.AddOrSetValue(nextToken, 1);

  FTotals.AddOrSetValue(ContextKey, FTotals.Items[ContextKey] + 1);

  if FGlobalCounts.TryGetValue(nextToken, current) then
    FGlobalCounts.AddOrSetValue(nextToken, current + 1)
  else
    FGlobalCounts.AddOrSetValue(nextToken, 1);
  Inc(FGlobalTotal);

  // O(1) ensure token registered
  EnsureTokenId(nextToken);
end;

function TNGramModel.GetAlphabetIndex(const token: string): Integer;
var
  i: Integer;
begin
  for i := 0 to FAlphabet.Count - 1 do
    if FAlphabet[i] = token then
      Exit(i);
  Result := -1;
end;

function TNGramModel.TryGetInner(const ContextKey: string;
  out inner: TDictionary<string, Integer>): Boolean;
begin
  Result := FCounts.TryGetValue(ContextKey, inner);
end;

function TNGramModel.SumCounts(inner: TDictionary<string, Integer>): Integer;
var
  kv: TPair<string, Integer>;
begin
  Result := 0;
  for kv in inner do
    Inc(Result, kv.Value);
end;

procedure TNGramModel.Train(const text: string);
var
  tokens, trainTokens: TArray<string>;
  i: Integer;
  ctxKey, nextTok: string;
begin
  tokens := Tokenize(text);
  trainTokens := InsertSentenceBoundaries(tokens);
  if Length(trainTokens) <= FOrder then
    Exit;

  for i := 0 to Length(trainTokens) - FOrder - 1 do
  begin
    ctxKey := ContextKeySlice(trainTokens, i, FOrder);
    nextTok := trainTokens[i + FOrder];
    AddContext(ctxKey, nextTok);
  end;
end;

function TNGramModel.ApplyTopP(const tokens: TList<string>;
  const weights: TList<Double>; TopP: Double; out outTokens: TList<string>;
  out outWeights: TList<Double>): Boolean;
var
  idx: TList<Integer>;
  i: Integer;
  total, cum: Double;
begin
  Result := False;
  outTokens := nil;
  outWeights := nil;

  if (TopP <= 0.0) or (TopP >= 1.0) or (tokens.Count = 0) or
    (weights.Count <> tokens.Count) then
    Exit;

  idx := TList<Integer>.Create;
  try
    for i := 0 to weights.Count - 1 do
      idx.Add(i);
    // sort by weight descending
    idx.Sort(TComparer<Integer>.Construct(
      function(const L, R: Integer): Integer
      begin
        if weights[L] > weights[R] then
          Result := -1
        else if weights[L] < weights[R] then
          Result := 1
        else
          Result := 0;
      end));

    total := 0.0;
    for i := 0 to weights.Count - 1 do
      total := total + weights[i];
    if total <= 0 then
      Exit;

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

function TNGramModel.SampleLaplace(inner: TDictionary<string, Integer>;
total: Integer; Temperature: Double; TopK: Integer; TopP: Double;
const prevToken: string): string;
var
  pairs: TList<TPair<string, Integer>>;
  kv: TPair<string, Integer>;
  comparer: IComparer<TPair<string, Integer>>;
  i, Count: Integer;
  invTemp: Double;
  tokens, tkCandidates: TList<string>;
  weights, wtCandidates: TList<Double>;
  sumW, draw, weight: Double;
begin
  if inner = nil then
  begin
    if FAlphabet.Count = 0 then
      Exit(' ');
    Exit(FAlphabet[Random(FAlphabet.Count)]);
  end;

  pairs := TList < TPair < string, Integer >>.Create;
  tokens := TList<string>.Create;
  weights := TList<Double>.Create;
  tkCandidates := nil;
  wtCandidates := nil;
  try
    for kv in inner do
      pairs.Add(kv);

    if TopK > 0 then
    begin
      comparer := TComparer < TPair < string, Integer >>.Construct(
        function(const L, R: TPair<string, Integer>): Integer
        begin
          Result := R.Value - L.Value;
        end);
      pairs.Sort(comparer);
      if TopK < pairs.Count then
        pairs.Count := TopK;
    end;

    invTemp := 1.0 / Max(Temperature, 1E-6);
    sumW := 0.0;

    for i := 0 to pairs.Count - 1 do
    begin
      Count := pairs[i].Value;
      weight := Power(Count + 1, invTemp);
      tokens.Add(pairs[i].key);
      weights.Add(weight);
      sumW := sumW + weight;
    end;

    // Apply heuristics to discourage consecutive punctuation and repetition
    AdjustWeightsForHeuristics(tokens, weights, prevToken);

    // Optional TopP nucleus filtering
    if ApplyTopP(tokens, weights, TopP, tkCandidates, wtCandidates) then
    begin
      tokens.Free;
      weights.Free;
      tokens := tkCandidates;
      weights := wtCandidates;
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
    pairs.Free;
    tokens.Free;
    weights.Free;
  end;
end;

function TNGramModel.SampleNextToken(const fullContextTokens: TArray<string>;
Temperature: Double; TopK: Integer; TopP: Double): string;
var
  inner, backoffInner: TDictionary<string, Integer>;
  ctxKey, backKey, k, prevToken: string;
  ctxLen, ctxTotal, backTotal: Integer;
begin
  // Determine previous token (last in context)
  if Length(fullContextTokens) > 0 then
    prevToken := fullContextTokens[Length(fullContextTokens) - 1]
  else
    prevToken := '';

  // Try longest context down to 1
  for ctxLen := FOrder downto 1 do
  begin
    ctxKey := ContextKey(fullContextTokens, ctxLen);
    if TryGetInner(ctxKey, inner) then
    begin
      ctxTotal := FTotals.Items[ctxKey];

      // Backoff context: use (ctxLen - 1) if available, else unigram
      backoffInner := nil;
      backTotal := 0;
      if ctxLen > 1 then
      begin
        backKey := ContextKey(fullContextTokens, ctxLen - 1);
        if TryGetInner(backKey, backoffInner) then
          backTotal := FTotals.Items[backKey];
      end;
      if (backoffInner = nil) then
      begin
        backoffInner := FGlobalCounts;
        backTotal := FGlobalTotal;
      end;

      case FSmoothing of
        smWittenBell:
          begin
            Result := SampleWittenBell(inner, ctxTotal, backoffInner, backTotal,
              Temperature, TopK, TopP, prevToken);
            Exit;
          end;
        smWittenBellFast:
          begin
            Result := SampleWittenBellFast(inner, ctxTotal, backoffInner,
              backTotal, Temperature, TopK, TopP, prevToken);
            Exit;
          end;
      else
        Result := SampleLaplace(inner, ctxTotal + FAlphabet.Count, Temperature,
          TopK, TopP, prevToken);
        Exit;
      end;
    end;
  end;

  // No context found — fallback to unigrams
  if FGlobalCounts.Count > 0 then
  begin
    Result := SampleLaplace(FGlobalCounts, FGlobalTotal + FAlphabet.Count,
      Temperature, TopK, TopP, prevToken);
    Exit;
  end;

  if FAlphabet.Count = 0 then
    Exit(' ');
  Result := FAlphabet[Random(FAlphabet.Count)];
end;

function TNGramModel.SampleWittenBell(ctxInner: TDictionary<string, Integer>;
ctxTotal: Integer; backoffInner: TDictionary<string, Integer>;
backoffTotal: Integer; Temperature: Double; TopK: Integer; TopP: Double;
const prevToken: string): string;
var
  allTokens: TList<string>;
  seen: TDictionary<string, Boolean>;
  kv: TPair<string, Integer>;
  i, U: Integer;
  invTemp: Double;
  backWeights: TList<Double>;
  sumBack: Double;
  combWeights: TList<Double>;
  sumComb, draw, p1, p2, lambda: Double;
  token: string;
  bc, bw, p: Double;
  indices: TList<Integer>;
  topSum, acc: Double;
  tkCandidates: TList<string>;
  wtCandidates: TList<Double>;

  function GetCount(d: TDictionary<string, Integer>; const k: string): Integer;
  var
    v: Integer;
  begin
    if (d <> nil) and d.TryGetValue(k, v) then
      Result := v
    else
      Result := 0;
  end;

begin
  if (ctxInner = nil) or (ctxInner.Count = 0) then
    Exit(SampleLaplace(backoffInner, backoffTotal, Temperature, TopK, TopP,
      prevToken));

  allTokens := TList<string>.Create;
  seen := TDictionary<string, Boolean>.Create;
  backWeights := TList<Double>.Create;
  combWeights := TList<Double>.Create;
  indices := nil;
  tkCandidates := nil;
  wtCandidates := nil;
  try
    for kv in ctxInner do
      if not seen.ContainsKey(kv.key) then
      begin
        seen.Add(kv.key, True);
        allTokens.Add(kv.key);
      end;

    for kv in backoffInner do
      if not seen.ContainsKey(kv.key) then
      begin
        seen.Add(kv.key, True);
        allTokens.Add(kv.key);
      end;

    invTemp := 1.0 / Max(Temperature, 1E-6);
    sumBack := 0.0;
    for i := 0 to allTokens.Count - 1 do
    begin
      token := allTokens[i];
      bc := GetCount(backoffInner, token);
      bw := Power(bc + 1, invTemp);
      backWeights.Add(bw);
      sumBack := sumBack + bw;
    end;
    if sumBack <= 0 then
      sumBack := 1.0;

    U := ctxInner.Count;
    if (ctxTotal <= 0) then
      ctxTotal := SumCounts(ctxInner);
    lambda := U / (ctxTotal + U);

    sumComb := 0.0;
    for i := 0 to allTokens.Count - 1 do
    begin
      token := allTokens[i];
      p1 := GetCount(ctxInner, token) / (ctxTotal + U);
      p2 := (backWeights[i] / sumBack);
      p := p1 + lambda * p2;
      combWeights.Add(p);
      sumComb := sumComb + p;
    end;

    // Apply heuristics to discourage consecutive punctuation and repetition
    AdjustWeightsForHeuristics(allTokens, combWeights, prevToken);

    if sumComb <= 0 then
      Exit(SampleLaplace(ctxInner, ctxTotal + U, Temperature, TopK, TopP,
        prevToken));

    if ApplyTopP(allTokens, combWeights, TopP, tkCandidates, wtCandidates) then
    begin
      allTokens.Free;
      combWeights.Free;
      allTokens := tkCandidates;
      combWeights := wtCandidates;
      sumComb := 0.0;
      for i := 0 to combWeights.Count - 1 do
        sumComb := sumComb + combWeights[i];
    end;

    if TopK > 0 then
    begin
      indices := TList<Integer>.Create;
      for i := 0 to combWeights.Count - 1 do
        indices.Add(i);
      indices.Sort(TComparer<Integer>.Construct(
        function(const L, R: Integer): Integer
        begin
          if combWeights[L] > combWeights[R] then
            Result := -1
          else if combWeights[L] < combWeights[R] then
            Result := 1
          else
            Result := 0;
        end));
      if TopK < indices.Count then
        indices.Count := TopK;

      topSum := 0.0;
      for i := 0 to indices.Count - 1 do
        topSum := topSum + combWeights[indices[i]];

      draw := Random * Max(topSum, 1E-9);
      acc := 0.0;
      for i := 0 to indices.Count - 1 do
      begin
        acc := acc + combWeights[indices[i]];
        if draw <= acc then
          Exit(allTokens[indices[i]]);
      end;
      Exit(allTokens[indices[indices.Count - 1]]);
    end
    else
    begin
      draw := Random * sumComb;
      acc := 0.0;
      for i := 0 to combWeights.Count - 1 do
      begin
        acc := acc + combWeights[i];
        if draw <= acc then
          Exit(allTokens[i]);
      end;
      Exit(allTokens[allTokens.Count - 1]);
    end;

  finally
    allTokens.Free;
    seen.Free;
    backWeights.Free;
    combWeights.Free;
    if indices <> nil then
      indices.Free;
  end;
end;

function TNGramModel.Generate(const seed: string; GenLength: Integer;
Temperature: Double; TopK: Integer; TopP: Double): string;
var
  seedArr: TArray<string>;
  outTokens: TList<string>;
  contextTail: TArray<string>;
  nextTok: string;
  i, baseCount: Integer;
begin
  if GenLength <= 0 then
    Exit('');
  seedArr := Tokenize(seed);
  if (FType = ngWord) and ((Length(seedArr) = 0) or (seedArr[0] <> BOS_TOKEN))
  then
    seedArr := InsertSentenceBoundaries(seedArr);

  outTokens := TList<string>.Create;
  try
    for i := 0 to Length(seedArr) - 1 do
      outTokens.Add(seedArr[i]);

    baseCount := outTokens.Count;
    while outTokens.Count < baseCount + GenLength do
    begin
      // Build only the last FOrder tokens
      var
      tailCount := Min(FOrder, outTokens.Count);
      SetLength(contextTail, tailCount);
      for i := 0 to tailCount - 1 do
        contextTail[i] := outTokens[outTokens.Count - tailCount + i];

      nextTok := SampleNextToken(contextTail, Temperature, TopK, TopP);
      outTokens.Add(nextTok);

      if (FType = ngWord) and (nextTok = EOS_TOKEN) then
        Break; // or inject BOS as shown earlier
    end;

    Result := Detokenize(outTokens.ToArray);
  finally
    outTokens.Free;
  end;
end;

function TNGramModel.SampleWittenBellFast
  (ctxInner: TDictionary<string, Integer>; ctxTotal: Integer;
backoffInner: TDictionary<string, Integer>; backoffTotal: Integer;
Temperature: Double; TopK: Integer; TopP: Double;
const prevToken: string): string;
var
  U: Integer;
  lambda, R, invTemp: Double;
  pick: string;
begin
  // If no context, fall back directly to backoff (Laplace) and pass prevToken
  if (ctxInner = nil) or (ctxInner.Count = 0) then
    Exit(SampleLaplace(backoffInner, backoffTotal, Temperature, TopK, TopP,
      prevToken));

  // Ensure we have a correct total
  if ctxTotal <= 0 then
    ctxTotal := SumCounts(ctxInner);

  // Witten–Bell mixture weight
  U := ctxInner.Count;
  lambda := U / (ctxTotal + U);
  invTemp := 1.0 / Max(Temperature, 1E-6);

  // Mixture sampling:
  // with probability (1 - lambda) sample from seen tokens, else from backoff
  R := Random;
  if R < (1.0 - lambda) then
  begin
    // Sample from seen tokens only using counts^1/T
    pick := SampleCounts(ctxInner, invTemp, TopK, TopP);
    if pick <> '' then
      Exit(pick);
    // If SampleCounts returns empty (shouldn’t for non-empty ctxInner), fall through
  end;

  // Backoff branch, pass prevToken to Laplace
  Result := SampleLaplace(backoffInner, backoffTotal, Temperature, TopK, TopP,
    prevToken);
end;

function TNGramModel.AlphabetSize: Integer;
begin
  Result := FAlphabet.Count;
end;

procedure TNGramModel.GetStats(out Contexts: Integer; out Alphabet: Integer;
out GlobalTotal: Integer);
begin
  Contexts := FCounts.Count;
  Alphabet := FAlphabet.Count;
  GlobalTotal := FGlobalTotal;
end;

function TNGramModel.GetStatsasString: string;
var
  Contexts: Integer;
  Alphabet: Integer;
  GlobalTotal: Integer;
begin

  Self.GetStats(Contexts, Alphabet, GlobalTotal);

  Result := 'Context=' + Contexts.ToString + '  Alphabet=' + Alphabet.ToString +
    '  GlobalTotal=' + GlobalTotal.ToString;

end;

function TNGramModel.EncodeB64(const s: string): string;
begin
  Result := TNetEncoding.Base64.Encode(s);
end;

function TNGramModel.EnsureTokenId(const token: string): Integer;
var
  id: Integer;
begin
  if not FTokenToIndex.TryGetValue(token, id) then
  begin
    id := FAlphabet.Count;
    FAlphabet.Add(token);
    FTokenToIndex.Add(token, id);
  end;
  Result := id;
end;

function TNGramModel.DecodeB64(const s: string): string;
begin
  Result := TNetEncoding.Base64.Decode(s);
end;

procedure TNGramModel.SaveToFile(const FileName: string);
var
  sl: TStringList;
  key, tok: string;
  inner: TDictionary<string, Integer>;
  sType, sPreserve, sSmoothing: string;
begin
  sl := TStringList.Create;
  try
    // Header
    sl.Add(FILE_MAGIC);

    // Model settings
    if FType = ngChar then
      sType := 'char'
    else
      sType := 'word';

    if FPreserveCase then
      sPreserve := '1'
    else
      sPreserve := '0';

    case FSmoothing of
      smLaplace:
        sSmoothing := 'laplace';
      smWittenBell:
        sSmoothing := 'wittenbell';
      smWittenBellFast:
        sSmoothing := 'wittenbellfast';
    else
      sSmoothing := 'laplace';
    end;

    sl.Add('order=' + IntToStr(FOrder));
    sl.Add('type=' + sType);
    sl.Add('preserveCase=' + sPreserve);
    sl.Add('smoothing=' + sSmoothing);

    // Alphabet
    sl.Add('alphabet_count=' + IntToStr(FAlphabet.Count));
    for tok in FAlphabet do
      sl.Add('alpha=' + EncodeB64(tok));

    // Global counts
    sl.Add('global_total=' + IntToStr(FGlobalTotal));
    for tok in FGlobalCounts.Keys do
      sl.Add('global=' + EncodeB64(tok) + #9 +
        IntToStr(FGlobalCounts.Items[tok]));

    // Context counts
    for key in FCounts.Keys do
    begin
      inner := FCounts.Items[key];
      for tok in inner.Keys do
        sl.Add('ctx=' + EncodeB64(key) + #9 + EncodeB64(tok) + #9 +
          IntToStr(inner.Items[tok]));
    end;

    sl.SaveToFile(FileName, TEncoding.UTF8);
  finally
    sl.Free;
  end;
end;

function TNGramModel.ContextKeySlice(const tokens: TArray<string>;
startIdx, Count: Integer): string;
var
  i: Integer;
  sb: TStringBuilder;
begin
  if Count <= 0 then
    Exit('');
  sb := TStringBuilder.Create;
  try
    for i := 0 to Count - 1 do
    begin
      sb.Append(tokens[startIdx + i]);
      if i < Count - 1 then
        sb.Append(CONTEXT_DELIM);
    end;
    Result := sb.ToString;
  finally
    sb.Free;
  end;
end;

procedure TNGramModel.LoadFromFile(const FileName: string);
var
  sl: TStringList;
  i: Integer;
  line, part, k, t: string;
  p1, p2: Integer;
  Value: Integer;
  inner: TDictionary<string, Integer>;

  function StartsWith(const s, pref: string): Boolean;
  begin
    Result := (Length(s) >= Length(pref)) and (Copy(s, 1, Length(pref)) = pref);
  end;

begin
  // Clear existing
  for k in FCounts.Keys do
  begin
    inner := FCounts.Items[k];
    inner.Free;
  end;
  FCounts.Clear;
  FTotals.Clear;
  FGlobalCounts.Clear;
  FGlobalTotal := 0;
  FAlphabet.Clear;

  sl := TStringList.Create;
  try
    sl.LoadFromFile(FileName, TEncoding.UTF8);
    if (sl.Count = 0) or (sl[0] <> FILE_MAGIC) then
      raise Exception.Create('Invalid model file or magic header');

    for i := 1 to sl.Count - 1 do
    begin
      line := sl[i];

      if StartsWith(line, 'order=') then
        FOrder := StrToInt(Copy(line, 7, MaxInt))
      else if StartsWith(line, 'type=') then
      begin
        part := Copy(line, 6, MaxInt);
        if part = 'char' then
          FType := ngChar
        else
          FType := ngWord;
      end
      else if StartsWith(line, 'preserveCase=') then
        FPreserveCase := Copy(line, 14, MaxInt) = '1'
      else if StartsWith(line, 'smoothing=') then
      begin
        part := Copy(line, 11, MaxInt);
        if part = 'wittenbell' then
          FSmoothing := smWittenBell
        else
          FSmoothing := smLaplace;
      end
      else if StartsWith(line, 'alphabet_count=') then
      begin
        // Count header; actual tokens follow in 'alpha=' lines
      end
      else if StartsWith(line, 'alpha=') then
      begin
        part := Copy(line, 7, MaxInt);
        FAlphabet.Add(DecodeB64(part));
      end
      else if StartsWith(line, 'global_total=') then
      begin
        FGlobalTotal := StrToInt(Copy(line, 14, MaxInt));
      end
      else if StartsWith(line, 'global=') then
      begin
        // global=<tok_b64>\t<count>
        p1 := Pos(#9, line);
        k := DecodeB64(Copy(line, 8, p1 - 8));
        Value := StrToInt(Copy(line, p1 + 1, MaxInt));
        FGlobalCounts.Add(k, Value);
      end
      else if StartsWith(line, 'ctx=') then
      begin
        // ctx=<ctx_b64>\t<tok_b64>\t<count>
        p1 := Pos(#9, line);
        p2 := PosEx(#9, line, p1 + 1);
        if (p1 > 0) and (p2 > p1) then
        begin
          k := DecodeB64(Copy(line, 5, p1 - 5));
          t := DecodeB64(Copy(line, p1 + 1, p2 - p1 - 1));
          Value := StrToInt(Copy(line, p2 + 1, MaxInt));

          if not FCounts.TryGetValue(k, inner) then
          begin // reconstruct inner map and total
            inner := TDictionary<string, Integer>.Create;
            FCounts.Add(k, inner);
            FTotals.Add(k, 0);
          end;

          inner.AddOrSetValue(t, Value);
          FTotals.AddOrSetValue(k, FTotals.Items[k] + Value);
          if GetAlphabetIndex(t) = -1 then
            FAlphabet.Add(t);
        end;
      end;
    end;

    // If global_total was missing, recompute
    if FGlobalTotal = 0 then
    begin
      for k in FGlobalCounts.Keys do
        Inc(FGlobalTotal, FGlobalCounts.Items[k]);
    end;
  finally
    sl.Free;
  end;
end;

end.
