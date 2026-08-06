# Pure Pursuit 곡률 추종 개선 보고서

작성일: 2026-08-06  
대상: `/home/tak/dolbatS/src/drive_pkg`

## 1. 결론

이번 검토에서 추천한 항목을 다음과 같이 구분했다.

| 구분 | 항목 | 처리 결과 |
|---|---|---|
| 즉시 수정 가능 | PP가 REF보다 부족한 경우에만 곡률 보정 | 적용 완료 |
| 즉시 수정 가능 | 같은 방향에서 `abs(PP) >= abs(REF)`이면 PP 조향 유지 | 적용 완료 |
| 즉시 수정 가능 | 미세한 곡률 부족분 deadband | `0.01 1/m` 적용 완료 |
| 즉시 수정 가능 | 적용/차단 이유와 부족분 상태값 발행 | 기존 상태 토픽 JSON에 추가 완료 |
| 즉시 수정 가능 | 좌/우, 상한, 반대 부호, fallback 회귀 시험 | 적용 및 통과 |
| 점증 검증 필요 | `steering_gain: 2.0 -> 1.0` | 실차 보정 전에는 적용하지 않음 |
| 점증 검증 필요 | `curvature_tracking_gain: 0.20 -> 1.0` | 단계별 실차 확인 전에는 적용하지 않음 |
| 실측 필요 | 조향각-ADC/PWM 좌우 비대칭 LUT | 측정 데이터가 없어 적용하지 않음 |
| 후속 제어 단계 | 횡오차 `e_y` 기반 P/PI feedback | one-sided 보정 실차 평가 후 검토 |

즉시 적용 항목은 기존 `pure_pursuit` 노드 안에서만 수정했다. 패키지 구조,
launch, 노드, 토픽 이름과 메시지 타입은 변경하지 않았다.

## 2. 수정 배경

직전 곡률 추종식은 다음과 같은 양방향 보정이었다.

```text
tracking_error = kappa_REF - kappa_PP
correction = gain * clip(tracking_error, -max_error, +max_error)
kappa_command = kappa_PP + correction
```

이 식은 같은 방향에서 PP가 REF보다 클 때 correction을 PP 반대 방향으로
만든다.

```text
kappa_PP  = +0.40 1/m
kappa_REF = +0.30 1/m
gain      = 0.20

tracking_error = -0.10 1/m
correction     = -0.02 1/m
kappa_command  = +0.38 1/m
```

Pure Pursuit의 큰 곡률에는 레퍼런스 경로 자체의 굽음뿐 아니라 차량을 경로로
복귀시키는 횡오차 피드백 성분이 포함될 수 있다. REF가 더 작다는 이유만으로
PP를 감소시키면 필요한 경로 복귀 조향을 약화할 수 있다.

## 3. 현재 적용한 one-sided 곡률 부족분 보정

현재 제어식은 다음과 같다.

```text
deficit = max(abs(kappa_REF) - abs(kappa_PP), 0)

bounded_deficit = min(
    deficit,
    curvature_tracking_max_correction_1pm
)

correction = sign(kappa_REF)
           * curvature_tracking_gain
           * bounded_deficit

kappa_command = kappa_PP + correction
```

단, 다음 품질·방향 보호 조건을 먼저 통과해야 한다.

```text
reference curvature valid
AND fallback path가 아님
AND abs(kappa_REF) >= sign_guard
AND 의미 있는 PP와 REF가 반대 방향이 아님
AND 같은 방향에서 PP가 이미 REF 이상이 아님
AND deficit > min_deficit
```

핵심 특성은 다음과 같다.

- REF는 PP에 부족한 같은 방향 곡률만 보충한다.
- 같은 방향에서 PP가 이미 REF 이상이면 PP를 절대 감소시키지 않는다.
- 좌/우 곡선을 동일한 식으로 처리한다.
- 최대 부족분 상한과 gain은 그대로 유지한다.
- 최종 각도 제한과 시간 기반 rate/acceleration limiter는 그대로 유지한다.

## 4. 분기별 현재 동작

| 조건 | correction | reason |
|---|---:|---|
| 기능 비활성화 | `0` | `disabled` |
| fallback 경로 | `0` | `fallback_path` |
| 곡률 표본 부족/불안정 | `0` | 해당 invalid reason |
| `abs(REF) < 0.05 1/m` | `0` | `reference_curvature_below_guard` |
| 의미 있는 PP와 REF 방향 반대 | `0` | `opposite_direction_guard` |
| 같은 방향이고 `abs(PP) >= abs(REF)` | `0` | `pp_already_sufficient` |
| 부족분 `<= 0.01 1/m` | `0` | `curvature_deficit_below_guard` |
| PP 곡률 부족 | REF 방향으로 제한 보충 | `applied` |

### 왼쪽 곡선 예

```text
kappa_PP  = +0.15
kappa_REF = +0.30
deficit   = 0.15
gain      = 0.20
correction = +0.03
kappa_command = +0.18
```

### 오른쪽 곡선 예

```text
kappa_PP  = -0.15
kappa_REF = -0.30
deficit   = 0.15
correction = -0.03
kappa_command = -0.18
```

### PP가 이미 충분한 예

```text
kappa_PP  = +0.40
kappa_REF = +0.30
correction = 0
kappa_command = +0.40
reason = pp_already_sufficient
```

## 5. 현재 REF 곡률 품질 판정

현재 코드는 과거 문서의 “목표점 전후 한 개 3점” 방식이 아니다.

- 목표점 주변부터 전방 `0.45 m`를 조사한다.
- 호길이 `0.15 m` 간격의 3점 원곡률을 반복 계산한다.
- 반복 시작점 간격은 `0.075 m`다.
- 최소 3개 유효 표본을 요구한다.
- 곡률 표본 중앙값을 REF로 사용한다.
- 곡률 MAD가 `0.25 1/m`를 넘으면
  `unstable_preview_curvature`로 보정을 차단한다.

따라서 경계 융합 이음부나 B-spline의 국소 꺾임 한 번이 곧바로 곡률 보정에
들어가는 것을 줄인다.

## 6. 현재 YAML 파라미터

위치:

```text
src/drive_pkg/config/drive_pipeline.yaml
pure_pursuit.ros__parameters
```

```yaml
minimum_path_preview_m: 0.30

curvature_tracking_enabled: true
curvature_tracking_gain: 0.20
curvature_tracking_sample_gap_m: 0.15
curvature_tracking_preview_m: 0.45
curvature_tracking_min_samples: 3
curvature_tracking_max_mad_1pm: 0.25
curvature_tracking_max_correction_1pm: 0.20
curvature_tracking_sign_guard_1pm: 0.05
curvature_tracking_min_deficit_1pm: 0.01
```

`curvature_tracking_max_correction_1pm`은 gain 적용 전 부족분 상한이다.
현재 최대 실제 correction은 다음과 같다.

```text
0.20 gain x 0.20 1/m = 0.04 1/m
```

현재 YAML의 `steering_gain=2.0`까지 고려하면 직진 근처에서 이 correction의
각도 영향은 최대 약 `2.5 deg` 수준이다. 기존 문서의 약 `1.25 deg`는
`steering_gain=1`을 전제로 한 값이므로 현재 설정 설명으로는 맞지 않는다.

## 7. 변경 파일

| 파일 | 이번 적용 내용 |
|---|---|
| `src/drive_pkg/pure_pursuit.py` | one-sided deficit 보정, PP 충분 조건, deadband, 상태값 추가 |
| `src/drive_pkg/config/drive_pipeline.yaml` | `curvature_tracking_min_deficit_1pm: 0.01` 추가 |
| `src/drive_pkg/test/test_dynamic_lookahead.py` | 좌/우 부족분, PP 충분, deadband, 상한, 반대 방향 시험 추가 |
| `CURVATURE_TRACKING_UPDATE.md` | 현재 구현과 검증 결과로 전면 갱신 |

시각화 노드, launch 및 토픽 구성은 변경하지 않았다.

## 8. `/lane/control/status` 진단값

새 토픽을 만들지 않고 기존 `std_msgs/String` JSON에 아래 값을 유지·추가한다.

| 키 | 의미 |
|---|---|
| `pure_pursuit_curvature_1pm` | 목표점으로 계산한 PP 곡률 |
| `target_path_curvature_1pm` | 품질 판정을 거친 REF 곡률 |
| `target_path_curvature_samples` | REF 계산에 사용한 표본 수 |
| `target_path_curvature_mad_1pm` | 곡률 표본 MAD |
| `curvature_tracking_error_1pm` | signed `REF - PP` 진단값 |
| `curvature_tracking_deficit_1pm` | `max(abs(REF)-abs(PP), 0)` |
| `curvature_tracking_correction_1pm` | 최종 곡률에 실제 추가한 값 |
| `curvature_tracking_applied` | 보정 적용 여부 |
| `curvature_tracking_reason` | 적용/차단 이유 |

JSON 키 추가만 있으므로 기존 토픽 이름, 타입 및 기존 구독 구조는 유지된다.

## 9. 0803 bag 오프라인 비교

원본 입력 bag:

```text
/home/tak/Desktop/bags/rosbag2_2026_08_03-14_53_47
```

이 bag을 현재 차선 파이프라인으로 처리해 저장한 `/lane/path` 749프레임에
직전 양방향 보정과 새 one-sided 보정을 비교했다.

### 새 reason 분포

| reason | 프레임 수 |
|---|---:|
| `applied` | 385 |
| `pp_already_sufficient` | 13 |
| `curvature_deficit_below_guard` | 1 |
| `reference_curvature_below_guard` | 104 |
| `opposite_direction_guard` | 32 |
| `unstable_preview_curvature` | 64 |
| `insufficient_target_coverage` | 91 |
| `insufficient_curvature_samples` | 30 |
| `fallback_path` | 28 |
| `invalid_path` | 1 |

### raw 조향 분포

| 절댓값 | 직전 양방향 | 새 one-sided |
|---|---:|---:|
| 중앙값 | `6.3299 deg` | `6.3299 deg` |
| p90 | `12.3707 deg` | `12.3707 deg` |
| p95 | `13.0526 deg` | `13.0526 deg` |
| 최대 | `19.0122 deg` | `19.0122 deg` |

전체 분포는 변하지 않아 조향을 일괄 증폭하지 않았다. 다만 PP가 이미 충분한
13프레임에서는 직전 보정이 PP를 감소시키던 동작을 제거했다.

```text
PP 충분 프레임에서 보존된 조향 절댓값
중앙값: 0.4715 deg
p90:    1.2039 deg
최대:   1.7195 deg
```

전체 749프레임 중 raw 조향 계산값이 수치적으로 달라진 프레임은 29개다.
13개는 `pp_already_sufficient`, 1개는 deficit deadband이며, 나머지는 sign
guard 아래의 매우 작은 반대 방향 PP를 one-sided 크기식으로 처리한 차이다.

rosbag은 open-loop이므로 이 결과는 제어식 회귀 검증이다. 실제 횡오차와
oscillation 개선 여부는 저속 실차에서 확인해야 한다.

## 10. 검증 결과

- Python 구문 검사: 통과
- `git diff --check`: 통과
- `drive_pkg` 단위 테스트: 50개 전부 통과
- 추가 시험:
  - 왼쪽 곡률 부족분 보충
  - 오른쪽 곡률 부족분 보충
  - 왼쪽 PP 충분 시 감소 금지
  - 오른쪽 PP 충분 시 감소 금지
  - 작은 deficit deadband
  - correction 상한
  - 반대 방향 guard 유지
  - fallback 보정 차단 유지
- 실제 ROS 2 `pure_pursuit` 노드 생성 및 새 YAML 로드: 성공
- 패키지 빌드: 수행하지 않음

단위 테스트 셸에서는 ROS 메시지 바인딩용 `/tmp` test stub을 사용했다. 별도로
실제 ROS 2 환경에서 노드 생성과 파라미터 로드를 확인했다.

## 11. 지금 적용하지 않은 점증 검증 항목

### 11.1 `steering_gain=1.0`

현재 `steering_gain=2.0`을 바로 `1.0`으로 바꾸지 않았다. one-sided 보정은
REF가 유효하고 PP보다 클 때만 작동하므로 다음 프레임은 gain 감소를 보상하지
못한다.

- 거의 직선이지만 차량이 경로에서 벗어난 프레임
- PP가 REF보다 큰 경로 복귀 프레임
- PP/REF 방향 충돌 프레임
- 곡률 표본 부족 또는 불안정 프레임
- fallback 프레임

동일 bag에서 `steering_gain=1`, one-sided gain `1.0`을 가정해도 raw 조향
p90은 약 `12.37 deg -> 10.20 deg`, 최대는 `19.01 deg -> 12.78 deg`로
감소했다. 749프레임 중 149프레임은 현재보다 3도 이상 작았다.

따라서 실제 바퀴각 보정 없이 `steering_gain=1`을 동시에 적용하면 안 된다.

### 11.2 Arduino 조향 calibration

현재 Serial Bridge는 `/auto_steer_angle`을 `S,<degree>` 형식으로 그대로 보내고,
Arduino는 하나의 선형 `DEG_PER_ADC`로 목표 ADC를 계산한다. 다음 값의 실측이
먼저 필요하다.

```text
명령 degree
Arduino 목표 ADC
현재 ADC
실제 앞바퀴 degree
좌/우 도달 시간
정상상태 오차
```

좌우 비대칭이나 비선형성이 확인되면 Arduino의 degree-to-ADC 변환을 실측 LUT
또는 좌우 개별 보정식으로 바꾼다. 이 작업은 센서 실측값 없이 추정해서 적용할
수 없다.

### 11.3 `curvature_tracking_gain` 점증

현재값 `0.20`을 유지했다. 다음 조건을 만족한 뒤 저속에서만 점증한다.

```text
0.20 -> 0.40 -> 0.60 -> 필요 시 0.80/1.00
```

각 단계에서 확인할 값:

- `target_path_curvature_mad_1pm`
- `curvature_tracking_reason`
- `raw_steering_deg`
- limiter 이후 `steering_deg`
- `/vehicle/angle`
- 실제 횡오차와 좌우 overshoot

### 11.4 횡오차 feedback

조향 calibration과 one-sided 보정 후에도 곡선 바깥으로 지속적으로 밀리면
REF gain을 무작정 키우지 않는다. 경로 곡률 feedforward와 별도로 차량 기준
횡오차 `e_y`를 정의하고 P 또는 저속 PI feedback을 제한적으로 추가한다.
이 단계는 실차 횡오차 로그가 확보된 뒤 진행한다.

## 12. 실행 및 적용 확인

현재 설치는 source-linked 개발 설치이므로 패키지 빌드 없이 Python 프로세스
재시작으로 반영된다. 기존 launch를 `Ctrl+C`로 종료한 뒤 실행한다.

```bash
cd /home/tak/dolbatS
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

ros2 launch drive_pkg drive_pipeline.launch.py \
  params_file:=/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

파라미터 확인:

```bash
ros2 param get /pure_pursuit curvature_tracking_min_deficit_1pm
```

기대값:

```text
Double value is: 0.01
```

상태 확인:

```bash
ros2 topic echo /lane/control/status
```

특히 다음 값을 함께 확인한다.

```text
pure_pursuit_curvature_1pm
target_path_curvature_1pm
curvature_tracking_deficit_1pm
curvature_tracking_correction_1pm
curvature_tracking_reason
raw_steering_deg
steering_deg
```

곡률 보정 전체를 비활성화해 기본 PP와 비교하려면:

```yaml
curvature_tracking_enabled: false
```

## 13. 삭제되거나 바뀌지 않은 항목

- 삭제된 패키지/노드/launch: 없음
- 삭제되거나 이름이 바뀐 토픽: 없음
- 변경된 메시지 타입: 없음
- 삭제된 기존 파라미터: 없음
- 삭제된 최대 조향각 제한: 없음
- 삭제된 steering rate/acceleration limiter: 없음
- `steering_gain=2.0`: 이번 수정에서 유지
- Arduino 코드: 이번 수정에서 변경하지 않음
- Mission Manager 코드/설정: 이번 수정에서 변경하지 않음
