# LMP 오픈소스 공개 전 정리 계획

작성일: 2026-09-30
브랜치: `cleanup/open-source` (기준 커밋 `b477b6f`, `fix/bigfiles-pr`와 동일)
작업 위치: `/home/choi/LMP-oss` (git worktree)

> 이 문서는 정리 작업용 내부 문서이며, 공개 직전에 삭제한다.

---

## 0. 작업 원칙

1. **실행 중인 실험을 건드리지 않는다.**
   `/home/choi/LMP/LMP`는 지금 GPU 실험 두 개가 `--project=LMP`로 불러 쓰고 있다.
   정리 작업은 모두 `/home/choi/LMP-oss` worktree에서만 한다.
2. **수치 결과가 바뀌면 안 된다.**
   리팩터링 전에 고정 시드로 forward/backward/loss 값을 저장해 두고(골든 테스트), 매 단계마다 비교한다.
3. **공개 저장소는 새 히스토리로 시작한다.**
   현재 `.git` 팩이 약 320MB이고, 대부분 노트북 출력과 `.jld2` 가중치다.
   기존 비공개 저장소(`qwermirmir/LMP`)는 연구용으로 그대로 두고, 정리가 끝난 트리를 새 저장소에 단일 초기 커밋으로 올린다.
   (대안: `git filter-repo`로 대용량 파일 제거. 커밋 123개 대부분이 `docs:` 계획 문서라 보존 가치가 낮다.)
4. **연구 스크립트와의 호환은 공개 브랜치의 목표가 아니다.**
   바깥 `scripts/` 203개 파일은 기존 브랜치로 계속 돌리고, 공개 API가 안정되면 점진적으로 옮긴다.

---

## 1. 현황 요약

| 항목 | 현재 상태 |
|---|---|
| src | 13개 파일, 약 22.8k줄 |
| export | 132개. 정의되지 않은 이름 약 10개, 내부용(`_to_cpu` 등) 포함 |
| 자동 테스트 | `runtests.jl` 없음. `@testset` 쓰는 파일 5개, 나머지 39개는 노트북 |
| docstring | 약 870개 정의 중 약 73개 (8%) |
| 레거시 코드 | Attention/PTC/CPTC/siteweight 등 약 3.5k줄. 현재 모델은 fermionfield, DenseLinear, PT, Norm, BlockScalar, Ritz 레이어만 사용 |
| 한글 주석/에러 메시지 | 약 344줄 |
| 디버그 출력 | `Layer_sequence.jl` println 85개, `DDalphaAMG.jl` 64개 |
| 추적 파일 | research/ 노트북·jld2 약 570MB, docs/superpowers AI 계획 문서 34개 |
| LICENSE / README | 없음 / 0바이트 |

---

## 2. 결정이 필요한 사항 (작업 전에 정할 것)

1. **공개 범위**
   - 권장: 핵심 LMP(레이어, 역전파, Adam, 학습 루프, 게이지 I/O)만 `LMP` 패키지로 공개.
   - `DDalphaAMG.jl`(5.1k줄, 독립적인 Wilson AMG 솔버)은 별도 패키지 또는 package extension으로 분리.
   - `CoarseCorrectionLoss`, `GDlambdaAMG`, `GDlambdaAMGDDalphaAdapter`, `SubspaceSimilarity`는 연구용이므로 제외하거나 `LMP.Experimental`로 격리.
   - Attention 계열 레이어를 논문 등에서 쓸 예정이면 남기고, 아니면 제거.
2. **라이선스**
   - 원본 DDalphaAMG(C, mrottmann)은 GPL-3.0이다.
   - `src/DDalphaAMG.jl`이 C 코드를 번역한 것이라면 파생물로 볼 수 있어 GPL-3.0을 따라야 한다. 이 경우 분리해서 GPL로 공개하고, 핵심 LMP는 MIT로 두는 구성이 깔끔하다.
   - 독자 구현이라면 전체를 MIT로 해도 된다.
3. **패키지 이름**
   - Julia General 레지스트리의 AutoMerge는 5자 미만 이름을 자동 승인하지 않는다. `LMP`로 등록하려면 수동 승인이 필요하다.
   - 등록할 계획이면 풀네임 계열 이름을 검토한다. 등록 없이 GitHub URL로만 배포하면 그대로 둬도 된다.
4. **CUDA 의존성**
   - 현재 타입이 `CuArray`를 하드코딩하고 있어 CUDA가 필수다.
   - v0.1에서는 필수 의존성으로 유지하고, CPU 전용 경로는 이후 과제로 미루는 것을 권장한다. 단, 테스트는 CPU에서 돌아가야 CI가 가능하다.

---

## 3. 단계별 계획

### Phase 0. 준비 (코드 변경 없음)

- [ ] 메인 worktree의 미커밋 변경(`path_variant` 키워드, 체크포인트 path 목록 저장)을 사용자가 `fix/bigfiles-pr`에 커밋한 뒤 이 브랜치로 가져온다. 바깥 스크립트가 이 기능에 의존한다.
- [x] 골든 테스트 데이터 생성: 4⁴ hot 게이지, 고정 시드로 `build_lmp_32pt_layers` 기반 소형 모델의 forward 출력, loss, gradient를 저장한다. CPU에서 돌아가야 한다.

### Phase 1. 저장소 위생 (가장 쉽고 효과 큼)

삭제:
- [x] `research/` 전체 (노트북 26개, `.jld2` 5개, `compare_lmp_to_low_modes.jl`)
- [x] `docs/superpowers/` (AI 계획·스펙 문서 34개, `/data1`·GPU 번호 하드코딩 포함)
- [x] 루트 노트북: `Hop_test`, `shift_test`, `Clifford_test`, `module_test` 삭제. `raw_gauge_converter`는 Phase 5에서 예제로 재작성한 뒤 삭제
- [x] `할일목록`, `.vscode/`, `.claude/settings.local.json` (Windows 사용자 경로, git push 권한 포함)
- [ ] `test/`의 노트북 중 변환하지 않을 것들 (목록은 Phase 2 참고)
  - 레거시 레이어용 10개(`Attention_gradient`, `FullAttention_gradient`, `ParallelTransportAttention_gradient`, `CPTC_delta`, `Dirac_gradient`, `covariant_test`, `gradient_checking`, `test_gradient`, `siteweight_Layergradient`, `minmaxNorm_delta`)는 결정 사항 1이 정해질 때까지 남겨 둠
- [x] `test/lmp_low_mode_experiment_config.jl`, `validate_external_coarse_vectors.jl`, `Attention_layer.jl`, 깨진 `Attention_gradient.jl`

정비:
- [x] `Manifest.toml` 추적 해제, `.gitignore` 정리 (현재 `/test/`, `*.jld2`가 무시 목록에 있는데 추적 중)
- [ ] `Project.toml`
  - [x] 제거: `IJulia`, `Revise`, `Wilsonloop`(미사용), `DisplayAs`, `Plots`(로드만 하고 호출 없음)
  - `PyPlot`: `train`이 라이브 matplotlib 창을 띄우는 부분을 콜백으로 바꾼 뒤 제거 (Phase 4)
  - [x] `TimerOutputs` compat `"0"` → 실제 버전 범위로
  - [ ] `authors`를 실제 이름으로 (사용자 확인 필요)
  - [x] `[extras]`/`[targets]`에 `Test` 추가
- [ ] `docs/lmp_layers.tex`는 유지하되 내부용 절("발표 시 강조점", "협업자용 서사", "현재 상태와 주의점")을 정리하고 구현 참조를 최종 src에 맞춘다. PDF는 추적하지 않고 CI나 릴리스에 첨부.

### Phase 2. 테스트 체계 구축 (리팩터링 전에 안전망)

- [x] `test/runtests.jl` 작성, CPU 전용 테스트와 GPU 테스트를 분리 (`CUDA.functional()`일 때만 GPU 테스트)
- [x] 기존 `@testset` 파일 정리
  - `subspace_similarity.jl`, `ddalphaamg_setup_modes.jl`, `ddalphaamg_setup_modes_gpu.jl`: 유지
  - `gdlambda_amg.jl`: 수치 테스트만 남기고, 소스 문자열 `occursin` 검사 약 78개와 저장소 밖 스크립트 참조를 제거
  - `adam_global_step.jl`: 소스 문자열 검사를 행동 검사로 교체
- [x] 노트북에서 자동 테스트로 변환 (모두 4⁴ CPU, 외부 데이터 불필요)
  - 결과: `test_clover.jl`, `test_gradients.jl`, `test_gauge_covariance.jl`(reverse_paths 포함). 변환한 노트북과 `hopping_covariance_test.jl`은 삭제
  1. `Clover_test.ipynb`: 이미 PASS/FAIL 기준이 있음. 최우선
  2. 유한차분 gradient 검사: `PTLayer_gradient`, `ferimon_layergradient`, `NormLayer_delta`, `Dirac_gradient`
     - 템플릿: 바깥 `scripts/diagnostics/check_corner33_three_layer_gradient.jl`
     - 이 스크립트가 CPU/GPU 타입 문제를 우회하려고 `Hopping_operator`를 재정의하는데, CPU 디스패치 버그로 보이므로 먼저 고친다
     - [x] 수정함: `PTLayer._hop_out`이 `Vector{Vector{WF3x1}}`(Union)이라 CPU 메서드에 맞지 않았음. `WF3x1CPU`로 고정. 골든 값 변화 없음. 바깥 스크립트의 재정의는 이제 불필요
  3. 게이지 공변성: `hopping_covariance_test.jl`(PASS/FAIL 출력 → `@test`), `covariant_test`, `hop_covariant`
  4. `reversepath`: `reverse_paths`가 역연산인지 검사
- [x] Phase 0의 골든 테스트 추가
- [x] GitHub Actions CI (CPU 테스트만, Julia 1.11). `.github/workflows/CI.yml`. GPU 없는 러너에서 실제로 돌려 보지는 않음

#### Phase 2에서 발견한 것

- **복소 gradient 규약**: `gradient`는 `g = ∂L/∂Re(w) − i ∂L/∂Im(w)`를 반환하고, Adam의 `complex(upd_re, -upd_im)`은 이 규약에서 올바른 하강 방향이다(유한차분으로 확인).
  예외는 `BlockMatrix_Layer`로, 켤레(`∂L/∂Re + i ∂L/∂Im`)를 반환하고 Adam도 `+upd_im` 커널을 따로 쓴다. 내부적으로는 일관되지만, Phase 4에서 규약을 하나로 통일하고 docstring에 명시한다.
  GPU 경로는 `test_gpu_consistency.jl`로 확인: 두 모델 모두 GPU의 loss, gradient 노름, Adam 1스텝 후 loss가 CPU와 약 1e-15로 일치한다. 규약 통일 작업 후에도 이 테스트로 검증할 수 있다.
- **Clover trace**: `compute_Fmunu`가 `(Q − Q†)/8`에서 trace를 빼지 않아 표준 SW 정의와 다르다. `test_clover.jl`에서 `@test_broken`으로 표시. 물리 결과가 바뀌므로 사용자 결정 필요.

### Phase 3. 모듈 구조와 공개 API

- [ ] `__precompile__(false)` 제거. 의존 패키지의 precompile을 막고 있다. `NVTX.activate()` 같은 로드 시점 부작용은 `__init__`으로 옮긴다.
- [ ] 모듈 경계 정리: `CoarseCorrectionLoss`, `GDlambdaAMG`, Adapter가 `LMP` 최상위에 그냥 include되어 이름공간을 공유한다 (`GDlambdaAMG`가 `CoarseCorrectionLoss`의 `_field_norm`에 의존). 모듈로 감싸거나 분리한다.
- [ ] export 정리
  - [x] 정의되지 않은 이름 제거: `BatchedFieldBuffer`, `materialize_fields`, `preD`, `doubletrain`, `HalfPAttention_Layer`, `ReLU`/`Sigmoid`/`Tanh` 등 19개(export 문 기준 21건). 사용자 코드에서 이미 `UndefVarError`이거나 Base/LatticeDiracOperators로 해석되던 이름이라 동작 변화 없음
  - [ ] 존재하지 않는 `RealAttention_Layer` import 제거
  - 내부용 제거: `_to_cpu`, `_to_gpu`, 디버그용 `Ref`
  - 충돌 위험 이름: `mean`(Statistics), `conj!`(Base와 별개 함수로 정의됨), `gradient`/`update!`/`train`/`Identity`(Flux·Zygote와 충돌). Base 확장으로 바꾸거나 이름을 바꾸거나 export에서 뺀다
  - `substitute_fermion!` 이중 정의 해소
- [ ] 공개 API 확정. 바깥 스크립트 사용 빈도가 실질적인 API다:
  `PTLayer`, `DenseLinearLayer`, `NormLayer`, `BlockScalar_Layer`, `Ritz_Layer`, `fermionfield_Layer`, `Adam`, `train`, `load_checkpoint!`, `refresh_model!`, `inner_product`, `norm2`, `Paths`, `build_lmp_32pt_layers`, `deflated_initial_guess`
- [ ] 스크립트에서 반복되는 비공개 호출을 공개 함수로 승격
  - `_make_raw_cpu_buffers` + `_make_gauge_staging` + `_read_raw_gauge_to_gpu!` (38회 반복) → `load_gauge!(U, path)`
  - `_default_lmp_paths(max_len; path_variant)` → 공개 path 빌더
  - `_to_gpu_fermions`, `_cvt_br` → CPU/GPU 전송 헬퍼
- [ ] `load_lmp_model`이 파일 이름을 정규식으로 파싱해 하이퍼파라미터를 추정하는 부분 → 체크포인트 안에 설정을 저장
- [ ] 이름 일관성: `elementry` → `elementary`, `PrallelTransportAttention_Layer` 오타, 레이어 이름 규칙 통일 (`PTLayer`/`NormLayer` 형태로). 파일명도 CamelCase로 통일

### Phase 4. 코드 정리

- [ ] **타입 파이러시 제거 (정확성 문제)**
  - [x] `elementry_base.jl:567` `Base.conj(WF3x1)`: 인자 타입이 없어 `Base.conj(::Any)`를 정의하고 있었음. `WF3x1CPU`로 제한
  - [x] `*(Number, Spin_matrix)` 이중 정의 제거 (나중 정의가 앞의 것을 덮어쓰고 있었음)
  - Gaugefields/LatticeDiracOperators 타입에 대한 `Base.+ - * /`, `minmax`, `broadcast!` 오버로드 → LMP 자체 함수나 래퍼 타입으로
    - 직접 스캔 결과 외부 함수 × 외부 타입 메서드 23개: `*` 7, `+` 4, `-` 4, `/` 1, `conj` 2, `broadcast!`, `minmax`, `ndims`, `deepcopy_internal`, `LatticeDiracOperators.substitute_fermion!` 각 1. 대부분 `elementry_base.jl`, 일부 `gpu.jl`
    - Aqua.jl `test_piracies`는 이것들을 잡지 못했음(하위 모듈 메서드를 보지 않는 것으로 보임). CI에 넣을 때 주의
  - `gpu.jl:1869` `deepcopy_internal`이 `fshifted`를 몰래 더미로 바꾸는 동작 → 명시적 복사 함수로
- [ ] `WF3x1CPU/GPU` 별칭이 8개 파일에 중복 정의 → 한 곳에서 정의
- [ ] 레거시 레이어 약 3.5k줄 제거 (결정 사항 1에 따라)
- [ ] 주석 처리된 코드 약 300줄, `if false` 블록 제거
- [ ] `train` 3개 메서드와 `train_timed` 통합. 공통 루프 하나에 옵션(타이밍, 로깅 콜백)으로
- [ ] `train`의 PyPlot 라이브 플롯 → 콜백 인자. 로그 파일도 기본으로 작업 디렉터리에 쓰지 않게
- [ ] `println` → `@debug`/`@info`, `verbose` 인자
- [ ] ENV 변수로 동작을 바꾸는 부분(`CoarseCorrectionLoss.jl:952`, `Layer_sequence.jl:3405`) → 함수 인자로
- [ ] 전역 포인터 캐시(`gpu.jl:1370`)가 `fields[1]`만 검증해 stale 포인터 위험 → 검증 강화 또는 명시적 캐시 객체
- [ ] `Layer_sequence.jl`(4.2k줄) 분할: forward, CPU/GPU 전송, 학습 루프, 게이지 I/O, 메모리 디버깅
- [ ] `activation.jl`은 실제로 역전파 코드이므로 이름 변경 (예: `backward.jl`)
- [ ] 한글 주석과 에러 메시지, 이모지를 영어로 (사용자에게 보이는 에러 메시지부터)

### Phase 5. 문서와 예제

- [ ] `README.md`: 무엇을 하는 패키지인지, 설치, 최소 예제, 인용 방법
- [ ] `LICENSE` (결정 사항 2)
- [ ] `CITATION.cff` (관련 논문이 있으면)
- [ ] 공개 API docstring (현재 레이어 struct는 하나도 문서화되어 있지 않음)
- [ ] `examples/`
  1. 소형 heatbath 게이지 생성 (바깥 `scripts/data/heatbath.jl` 축소판, 8⁴)
  2. 소형 학습 (`train_lmp.jl` 축소판, ENV 설정 약 40개 → 키워드 인자)
  3. 학습된 모델로 솔버 가속 효과 확인
  4. ILDG → 내부 포맷 변환 (`raw_gauge_converter.ipynb` 재작성)
  - 외부 데이터 경로는 인자나 환경 변수로 받고 `/data1` 하드코딩 금지
- [ ] `.gbin` 포맷 명세 문서화, 읽기 함수가 CUDA 없이도 동작하게
- [ ] (선택) Documenter.jl 문서 사이트

### Phase 6. 공개

- [ ] 최종 점검: `git grep`으로 `/data1`, `/home/choi`, `C:\Users`, 한글, 개인 정보 확인
- [ ] 이 문서(`CLEANUP_PLAN.md`) 삭제
- [ ] 새 GitHub 저장소에 단일 초기 커밋으로 push, v0.1.0 태그
- [ ] (선택) General 레지스트리 등록, Zenodo DOI

---

## 4. 권장 진행 순서와 규모 감

| 순서 | 단계 | 위험도 | 비고 |
|---|---|---|---|
| 1 | Phase 0, 1 | 낮음 | 삭제 위주. 반나절 |
| 2 | Phase 2 | 낮음 | 이후 모든 작업의 안전망 |
| 3 | Phase 4의 타입 파이러시, 레거시 제거 | 중간 | 골든 테스트로 확인 |
| 4 | Phase 3 | 중간 | 바깥 스크립트와 호환 끊김 |
| 5 | Phase 4 나머지 | 중간 | `train` 통합이 가장 큼 |
| 6 | Phase 5, 6 | 낮음 | |

각 Phase는 별도 커밋(또는 PR)으로 나누고, 매번 `runtests.jl`과 골든 테스트를 통과시킨다.
