unit GUI.FMX.TestLLMs;

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes,
  System.Variants,  System.Diagnostics ,
  System.IOUtils, System.StrUtils,
  FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics, FMX.Dialogs, FMX.Memo.Types,
  FMX.Controls.Presentation, FMX.Edit, FMX.ListBox,
  FMX.ScrollBox, FMX.Memo, FMX.StdCtrls, FMX.Objects,  math,
  ///
  /// -----
  ///
  Unit_Math.AI.Ngram,
  Unit_Math.AI.Ngram.helpers;

type
  /// <summary>
  ///   as of today this application only is a implementation of a NGRAM
  ///   algorithm . LLM my come later
  /// </summary>
  TMainFormLLMEvaluation = class(TForm)
    rctngl1: TRectangle;
    btn_LoadTraingData: TCornerButton;
    btn_TrainModel: TCornerButton;
    btn_SaveModel: TCornerButton;
    btn_LoadModel: TCornerButton;
    dlgOpenfile: TOpenDialog;
    dlgSavefile: TSaveDialog;
    pnl_statusbar: TPanel;
    lbl_StatusBar: TLabel;
    btn_CreateModel: TCornerButton;
    edt_NGramSize: TEdit;
    pnl_executeModel: TPanel;
    pnl_Input: TPanel;
    mmo_Inputtext: TMemo;
    pnl_Parameters: TPanel;
    edt_length: TEdit;
    edt_temperature: TEdit;
    edt_TopK: TEdit;
    edt_TopP: TEdit;
    btn_Execute: TButton;
    pnl_outdata: TPanel;
    mmo_Out: TMemo;
    lbl1: TLabel;
    lbl2: TLabel;
    lbl_TopK: TLabel;
    lbl_TopP: TLabel;
    lbl_Temperature: TLabel;
    lbl_Genlength: TLabel;
    cb_GramType: TComboBox;
    cb_Smoothing: TComboBox;
    chk_PreserveCase: TCheckBox;
    btn_SelectFolder: TCornerButton;
    dlgOpenFolder: TOpenDialog;
    procedure btn_LoadTraingDataClick(Sender: TObject);
    procedure btn_TrainModelClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure btn_ExecuteClick(Sender: TObject);
    procedure btn_SaveModelClick(Sender: TObject);
    procedure btn_LoadModelClick(Sender: TObject);
    procedure btn_CreateModelClick(Sender: TObject);
    procedure btn_SelectFolderClick(Sender: TObject);
  private
    { Private declarations }

    FModel: TNGramModel;

    FNGramSize: Integer;

    FTrainText, FOutput: string;

    FCorpusFiles: TStringlist;

    procedure UpdateStatus(Info: String);
    procedure AddFilesToCorpus(const Files: TArray<string>);
    procedure ValidateModelCreated;
    procedure TrainOverCorpus;

  public
    { Public declarations }
  end;

var
  MainFormLLMEvaluation: TMainFormLLMEvaluation;

implementation

{$R *.fmx}

procedure TMainFormLLMEvaluation.AddFilesToCorpus(const Files: TArray<string>);
var
  f: string;
begin
  for f in Files do
  begin
    if (FCorpusFiles.IndexOf(f) = -1) then
    begin
      FCorpusFiles.Add(f);

    end;
  end;
  UpdateStatus(Format('Corpus size: %d files', [FCorpusFiles.Count]));
end;

procedure TMainFormLLMEvaluation.btn_CreateModelClick(Sender: TObject);
var
  order: Integer;
  gramType: TNGramType;
  smoothing: TSmoothing;
  preserve: Boolean;
begin
  // Free previous model if any
  if FModel <> nil then
    FreeAndNil(FModel);

  order := StrToIntDef(edt_NGramSize.Text, 3);
  if order < 1 then
    order := 3;

  if cb_GramType.ItemIndex = 1 then
    gramType := ngChar
  else
    gramType := ngWord;

  case cb_Smoothing.ItemIndex of
    0:
      smoothing := smLaplace;
    1:
      smoothing := smWittenBell;
    2:
      smoothing := smWittenBellFast;
  end;

  preserve := chk_PreserveCase.IsChecked;

  // FModel := TNGramModel.Create(5, ngWord); // try orders 3..6 for speed and coherence
  FModel := TNGramModel.Create(order, gramType, preserve, smoothing);

  UpdateStatus(Format('Model created: order=%d, type=%s, smoothing=%s',
    [order, IfThen(gramType = ngWord, 'word', 'char'),
    IfThen(smoothing = smLaplace, 'Laplace', IfThen(smoothing = smWittenBell,
    'WittenBell', 'WittenBellFast'))]));
end;

procedure TMainFormLLMEvaluation.btn_ExecuteClick(Sender: TObject);
var
  inputstr: string;
  Temperature: Double;
  ExpectedLength: Integer;
  TopK: Integer;
  TopP: Double;
  sw: TStopwatch;
begin
  inputstr := Trim(mmo_Inputtext.Text);
  Temperature := ParseFloatAnyFormat(Trim(edt_temperature.Text), 0.0);
  TopP := ParseFloatAnyFormat(Trim(edt_TopP.Text), 1.0);
  TopK := StrToIntDef(Trim(edt_TopK.Text), 0);
  ExpectedLength := StrToIntDef(edt_length.Text, 20);

  mmo_Out.Lines.Clear;

  sw := TStopwatch.StartNew;
  FOutput := FModel.Generate(inputstr, ExpectedLength, Temperature, TopK, TopP);
  sw.Stop;

  mmo_Out.Lines.Add('[IN] =' + inputstr);
  mmo_Out.Lines.Add('[OUT]=' + FOutput);
  mmo_Out.Lines.Add('[PARAMETER]=' + IntToStr(ExpectedLength) + '/' +
    FloatToStr(Temperature) + '/' + FloatToStr(TopP) + '/' + IntToStr(TopK));
  mmo_Out.Lines.Add(Format('Time=%.3fs  Tokens/sec=%.1f',
    [sw.Elapsed.TotalSeconds, ExpectedLength / Max(0.001, sw.Elapsed.TotalSeconds)]));
end;


procedure TMainFormLLMEvaluation.UpdateStatus(Info: String);
begin
  lbl_StatusBar.Text := Info;

  Application.ProcessMessages;
end;

procedure TMainFormLLMEvaluation.btn_LoadModelClick(Sender: TObject);
begin
  if dlgOpenfile.Execute then
  begin
    FModel.LoadFromFile(dlgOpenfile.FileName);

    UpdateStatus('load model from file ');
  end;
end;

procedure TMainFormLLMEvaluation.btn_LoadTraingDataClick(Sender: TObject);
begin

  UpdateStatus('select *.txt  trainings files  ');

  if dlgOpenfile.Execute then
    AddFilesToCorpus(dlgOpenfile.Files.ToStringArray);

end;

procedure TMainFormLLMEvaluation.btn_SaveModelClick(Sender: TObject);
begin

  if dlgSavefile.Execute then
  begin
    FModel.SaveToFile(dlgSavefile.FileName);

    UpdateStatus('save model to file ');
  end;

end;

procedure TMainFormLLMEvaluation.btn_SelectFolderClick(Sender: TObject);
var
  Files: TArray<string>;
begin

    UpdateStatus('select all *.txt  inside a  folder ');

  if dlgOpenFolder.Execute then
  begin
    Files := TDirectory.GetFiles(dlgOpenFolder.FileName, '*.txt',
      TSearchOption.soAllDirectories);
    if Length(Files) = 0 then
      UpdateStatus('No .txt files found in the selected folder.')
    else
      AddFilesToCorpus(Files);
  end;
end;

procedure TMainFormLLMEvaluation.TrainOverCorpus;
var
  i: Integer;
  Text: string;
  ctx, alpha, total: Integer;
begin
  ValidateModelCreated;
  if FCorpusFiles.Count = 0 then
    raise Exception.Create
      ('Add files or a folder to the corpus before training.');

  mmo_Out.lines.Add('--- Training over corpus ---');
  for i := 0 to FCorpusFiles.Count - 1 do
  begin
    UpdateStatus(Format('Training (%d/%d): %s', [i + 1, FCorpusFiles.Count,
      ExtractFileName(FCorpusFiles[i])]));
    Text := TFile.ReadAllText(FCorpusFiles[i], TEncoding.UTF8);
    if Text.Trim <> '' then
    begin
      FModel.Train(Text);
      FModel.GetStats(ctx, alpha, total);
      mmo_Out.lines.Add
        (Format('Trained: %s | Contexts=%d Alphabet=%d GlobalTotal=%d',
        [ExtractFileName(FCorpusFiles[i]), ctx, alpha, total]));
    end
    else
      mmo_Out.lines.Add('Skipped empty file: ' + FCorpusFiles[i]);
  end;

  mmo_Out.lines.Add('--- After training ---');
  FModel.ComputeSparsityReport(mmo_Out.lines);
  UpdateStatus('Training complete.');
end;

procedure TMainFormLLMEvaluation.btn_TrainModelClick(Sender: TObject);
begin
  try
    TrainOverCorpus;
  except
    on E: Exception do
      UpdateStatus('Training failed: ' + E.Message);
  end;

end;

procedure TMainFormLLMEvaluation.ValidateModelCreated;
begin
  if FModel = nil then
    raise Exception.Create
      ('Create the model first (order, type, smoothing) before training.');
end;

procedure TMainFormLLMEvaluation.FormCreate(Sender: TObject);
begin

  Randomize;

  FCorpusFiles := TStringlist.Create;

  cb_GramType.Items.Clear;
  cb_GramType.Items.Add('word');
  cb_GramType.Items.Add('char');
  cb_GramType.ItemIndex := 0;

  cb_Smoothing.Items.Clear;
  cb_Smoothing.Items.Add('Laplace');
  cb_Smoothing.Items.Add('WittenBell');
  cb_Smoothing.Items.Add('WittenBellFast');
  cb_Smoothing.ItemIndex := 0;

  // Configure dialogs
  dlgOpenfile.Options := dlgOpenfile.Options + [TOpenOption.ofAllowMultiSelect];


   UpdateStatus('Welcome tiny NGRAM, (LLM) Evaluation ');

end;

end.
