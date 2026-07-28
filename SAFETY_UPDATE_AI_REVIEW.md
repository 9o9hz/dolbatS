# 자율주행 안전 변경 검토 보고서

## 1. 보고서 목적

이 문서는 `original_before_safety_update/`에 보존된 코드와 이후 적용됐던
안전 변경 코드를 비교하여, AI 에이전트가 안전 변경의 재도입 여부와 구현
방식을 점검할 수 있도록 작성한 검토 자료다.

현재 실행용 `src/drive_pkg`의 아래 6개 파일은 사용자의 요청에 따라
안전 변경 직전 보존본으로 복원된 상태다.

- `lane_processing.py`
- `path_plan.py`
- `pure_pursuit.py`
- `path_visualizer.py`
- `config/drive_pipeline.yaml`
- `launch/drive_pipeline.launch.py`

패키지 빌드와 실차 시험은 수행하지 않았다.

## 2. 현재 복원 상태의 핵심

현재 복원된 구조는 다음과 같다.

```text
/lane/detection/mask/compressed
          |
          v
      path_plan
       |      \
       |       +--> /lane/path/status (String, 경로 상태)
       v
 /lane/path (nav_msgs/Path, 경로 좌표)
          |
          v
    pure_pursuit
          |
          v
      /cmd_vel
```

`pure_pursuit`는 경로 좌표와 fallback 상태를 서로 다른 토픽에서 받는다.
두 메시지가 같은 카메라 프레임에서 만들어졌다는 것을 보장하는 식별자가
없으며, 수신 시각을 기준으로 watchdog을 갱신한다.

또한 복원된 YAML의 `pure_pursuit.enable_drive`는 `true`다. 따라서 실제
차량과 `/cmd_vel` 구독기가 연결돼 있다면 launch 직후 구동 명령이 전달될
수 있다. 실차 적용 전에는 반드시 `false`로 바꾸는 것이 안전하다.

## 3. 안전 변경이 이뤄진 이유

안전 변경의 주된 목적은 다음 세 가지 위험을 줄이는 것이었다.

### 3.1 오래된 경로를 정상 경로로 오인할 위험

기존 제어기는 `nav_msgs/Path`를 받은 시각만 저장한다. 경로의 실제 촬영
시각이 오래됐더라도 방금 수신됐다면 최신 경로처럼 처리할 수 있다.
연산 지연, ROS 큐 적체, rosbag 재생 또는 시스템 시간 불일치가 있을 때
이미 유효하지 않은 경로로 조향할 가능성이 있다.

### 3.2 fallback 경로가 예상보다 오래 유지될 위험

기존 `max_missing_frames`는 시간 대신 프레임 개수로 이전 경로 사용을
제한한다. 처리 속도가 30 FPS일 때 8프레임과 5 FPS일 때 8프레임의 실제
지속 시간은 크게 다르다. 카메라나 추론 속도가 느려지면 이전 경로가
의도보다 오래 제어에 사용될 수 있다.

### 3.3 경로와 상태 토픽이 서로 다른 프레임일 위험

기존에는 `/lane/path`와 `/lane/path/status`가 별도 메시지다. 콜백 실행
순서나 큐 지연에 따라 새 경로와 이전 프레임의 fallback 상태가 조합될
수 있다. 제어에 필요한 값들을 한 메시지로 묶으려는 이유가 여기에 있다.

## 4. 안전 변경에서 달라졌던 항목

### 4.1 `lane_processing.py`

- `max_fallback_sec` 파라미터를 추가했다.
- 마지막 정상 경로 생성 시각을 `time.monotonic()`으로 저장했다.
- 추적 경계 또는 이전 경로를 사용하는 동안 fallback 경과 시간을
  계산했다.
- 제한 시간을 넘으면 저장된 경로를 제거하고
  `reason="fallback_timeout"`으로 경로를 무효화했다.
- 처리 결과에 `fallback_age_sec`를 추가했다.

### 4.2 `path_plan.py`

- 제어 전용 `/lane/path/control` 토픽을 추가했다.
- 하나의 JSON 메시지에 아래 값을 함께 넣었다.

  - 원본 이미지의 `timestamp_ns`
  - `path_valid`
  - `reason`
  - `fallback`
  - `fallback_age_sec`
  - 미터 단위 `path_meters`

- 경로 계획 예외가 발생했을 때도 명시적인 무효 상태와 빈 제어 경로를
  발행하도록 변경했다.
- 기존 `/lane/path`는 시각화 및 호환 목적으로 계속 발행했다.

### 4.3 `pure_pursuit.py`

- 제어 입력을 `/lane/path`와 `/lane/path/status` 두 토픽에서
  `/lane/path/control` 단일 토픽으로 변경했다.
- 다음 검사를 추가했다.

  - 필수 필드와 배열 형식 검사
  - 빈 경로와 NaN/Inf 검사
  - 경로 촬영 시각 누락 검사
  - 너무 오래된 경로 검사
  - 미래 시각 경로 검사
  - fallback 절대 지속 시간 검사
  - 무수신 watchdog 검사

- 검사 실패 시 즉시 정지 명령을 발행하고 거부 이유를 상태 토픽에
  기록했다.
- 계산 조향각, 실제 명령 조향각, 경로 나이, fallback 나이를 구분해
  상태로 발행했다.

### 4.4 `path_visualizer.py`

- `/vehicle/current_steering_angle`을 구독해 차량이 보고한 실제 조향각을
  표시했다.
- 계산 조향각, 실제 명령 조향각, 실제 피드백 조향각을 구분했다.
- 경로와 제어 상태의 timestamp가 일치할 때만 Look-ahead 목표점을
  연결했다.
- 경로가 오래됐을 때 바로 숨기지 않고 `STALE`로 표시했다.
- 제어 정상, dry-run, 정지, 거부 사유, 경로 나이와 fallback 나이를
  디버그 패널에 표시했다.

실제 조향각 표시는 모니터링 기능이며, 조향 오차를 제어에 다시 반영하는
폐루프 제어 기능은 아니다.

### 4.5 YAML 및 launch

- 실차 자동 구동 기본값을 `enable_drive: true`에서 `false`로 변경했다.
- 아래 시간 제한 파라미터를 추가했다.

  - `max_path_age_sec: 0.40`
  - `max_future_path_sec: 0.10`
  - `max_fallback_sec: 0.45`
  - `path_timeout_sec: 0.50`

- launch에 `use_sim_time` 인자를 추가하고 모든 파이프라인 노드에 동일하게
  전달했다.
- rosbag 사용 시 `--clock`과 `use_sim_time:=true`를 함께 사용할 수 있게
  했다.

## 5. 안전 변경의 장점

1. 차선 인식이 멈추거나 느려졌을 때 이전 경로로 계속 주행하는 시간을
   실제 초 단위로 제한할 수 있다.
2. 경로 좌표와 fallback 상태가 한 메시지에 들어가므로 서로 다른 프레임의
   데이터가 잘못 결합될 가능성이 줄어든다.
3. 촬영 시각과 현재 ROS 시간을 비교해 지연된 경로를 거부할 수 있다.
4. 잘못된 JSON, 비정상 좌표, 미래 timestamp, 경로 계획 예외가 정지로
   연결되는 fail-safe 동작을 갖는다.
5. `enable_drive: false`가 기본이므로 설정 실수로 차량이 즉시 움직일
   가능성이 줄어든다.
6. 계산값, 명령값, 차량 피드백을 분리 표시하므로 조향 문제의 위치를
   진단하기 쉽다.
7. rosbag과 실시간 실행의 시간 기준을 명시적으로 맞출 수 있다.

## 6. 안전 변경의 단점과 주의점

1. `/lane/path/control`이 표준 ROS 메시지가 아닌 `std_msgs/String` JSON이다.
   스키마가 컴파일 시점에 검증되지 않고, 매 프레임 직렬화 비용이 발생한다.
   장기적으로는 timestamp, fallback, path를 포함하는 전용 ROS 인터페이스가
   더 적절하다.
2. 제어 토픽의 기본 reliable queue depth가 10이면 연산 부하 상황에서
   오래된 메시지가 큐에 쌓일 수 있다. timestamp 검사가 최종 방어를 하지만,
   제어 입력에는 `KEEP_LAST(1)` 등 저지연 QoS도 검토해야 한다.
3. `max_path_age_sec=0.40` 같은 고정 임계값은 카메라 FPS, YOLO 추론 시간,
   CPU 성능에 따라 정상 경로까지 거부할 수 있다.
4. live 실행에서 센서 timestamp와 ROS clock의 기준이 다르면 모든 경로가
   stale 또는 future로 판정될 수 있다.
5. rosbag을 `--clock` 없이 실행하거나 launch의 `use_sim_time` 설정을
   잘못 지정하면 경로가 계속 거부될 수 있다.
6. `lane_processing`은 fallback 지속 시간을 wall monotonic time으로
   계산하고, 제어기는 ROS time을 사용한다. rosbag pause, 배속 재생,
   시뮬레이션 환경에서는 두 시간축의 의미가 달라질 수 있다.
7. 최초 정상 경로 없이 fallback이 발생하면 내부 값이 무한대가 될 수 있다.
   Python JSON의 `Infinity`는 엄격한 JSON 표준과 호환되지 않는다. 발행 전에
   유한한 최대값으로 바꾸거나 즉시 무효 경로를 발행하는 편이 안전하다.
8. 시각화에서 오래된 경로를 계속 그리며 `STALE`만 표시하면 운전자가 이를
   현재 유효 경로로 오인할 수 있다. 색상 변경, 큰 경고 또는 경로 숨김 정책을
   함께 검토해야 한다.
9. 실제 조향 피드백은 표시만 하므로 명령과 실제 값의 큰 오차가 발생해도
   자동 정지하지 않는다.
10. 안전 검사 추가로 코드와 상태 종류가 늘어나므로 단위 테스트와 통합
    테스트 없이 파라미터만 조정하면 원인 파악이 어려워질 수 있다.

## 7. 복원으로 인해 현재 제거된 보호 기능

보존본으로 되돌리면서 실행 코드에서는 다음 보호 기능이 제거됐다.

- fallback의 초 단위 절대 제한
- 경로 촬영 시각 기반 stale/future 검사
- 경로와 제어 상태의 원자적 전달
- 경로 계획 예외의 명시적 제어 무효 메시지
- 실제 조향각 피드백 표시
- 계산 조향각과 실제 명령 조향각의 분리 표시
- 모든 노드의 `use_sim_time` launch 연동
- 기본 dry-run 설정

특히 현재 YAML은 `enable_drive: true`이므로 정지 상태에서 먼저 검증하지
않고 실차 launch를 실행하는 것은 권장하지 않는다.

## 8. 현재 문서 불일치

복원 후 `howtorun.md` 두 파일에는 안전 변경 구조가 일부 남아 있다.

- 루트 `howtorun.md`
- `src/drive_pkg/howtorun.md`

이 문서들은 `/lane/path/control`과 `/vehicle/current_steering_angle`을
현재 파이프라인의 구성으로 설명하지만, 복원된 `drive_pkg` 실행 코드는
각 토픽을 사용하지 않는다. AI 에이전트는 안전 변경의 최종 채택 여부를
결정한 뒤 문서도 같은 구조로 동기화해야 한다.

## 9. AI 에이전트 점검 요청 사항

AI 에이전트는 다음 순서로 검토한다.

1. 현재 복원본에서 경로와 상태가 서로 다른 프레임으로 결합될 수 있는
   실제 callback/QoS 시나리오를 분석한다.
2. `max_missing_frames`만으로 fallback 지속 시간을 제한하는 것이 목표
   카메라 FPS와 최저 처리 FPS에서 몇 초가 되는지 계산한다.
3. 안전 변경의 `/lane/path/control` JSON을 그대로 쓸지, 전용 ROS msg로
   바꿀지 판단한다.
4. 제어 토픽 QoS를 reliable depth 10, reliable depth 1, best-effort
   depth 1 중 어떤 정책으로 사용할지 근거와 함께 제안한다.
5. 카메라 header timestamp, ROS system time, `/clock`, rosbag pause 및
   배속 재생에서 시간 검사가 올바른지 검증한다.
6. fallback 시간 계산을 monotonic time과 ROS time 중 하나로 통일할지
   판단한다.
7. stale, future, fallback timeout, 무수신, JSON 오류마다 차량이 한 번이
   아니라 지속적으로 정지 명령을 받는지 확인한다.
8. actual steering 피드백의 단위, 부호, 조향 중앙값과 timeout을
   `control_pkg/serial_bridge.py` 기준으로 검증한다.
9. 안전 기능을 작은 단위로 재도입할 순서와 각 단계의 회귀 테스트를
   제시한다.
10. 최종적으로 실차 기본값은 `enable_drive: false`를 유지할지 판단한다.

## 10. 권장 테스트 시나리오

| 시나리오 | 기대 결과 |
|---|---|
| 정상 30 FPS 경로 수신 | 연속 제어, stale/fallback 경고 없음 |
| YOLO 추론 정지 | 설정된 시간 안에 `/cmd_vel` 정지 |
| 동일 fallback 경로 반복 수신 | 절대 제한 시간 이후 정지 |
| 1초 이상 오래된 timestamp | 즉시 stale 거부 및 정지 |
| 미래 timestamp | 즉시 future 거부 및 정지 |
| 빈 경로 또는 NaN 좌표 | 즉시 무효 처리 및 정지 |
| 경로 계획 예외 | 빈 경로와 명시적 오류 상태 발행 |
| ROS 토픽 큐 적체 | 최신 메시지만 사용하고 오래된 제어 폐기 |
| rosbag pause | 차량 명령 정지, 시각화는 명확한 STALE 표시 |
| rosbag 배속 재생 | ROS time 기준 임계값이 일관되게 동작 |
| 실제 조향 피드백 중단 | 표시가 timeout 후 N/A로 전환 |
| `enable_drive: false` | 계산은 표시하되 실제 속도·조향 명령은 0 |

## 11. 검토 시 우선 결론

안전 변경이 해결하려던 문제는 실차 자율주행에서 유효한 위험이다. 특히
오래된 경로 차단, fallback 시간 제한, 기본 dry-run은 재도입 우선순위가
높다. 다만 JSON String 제어 메시지, 서로 다른 시간축 사용, QoS depth 10은
그대로 확정하기 전에 개선 검토가 필요하다.

권장 방향은 다음과 같다.

1. 우선 `enable_drive: false`와 시간 기반 정지 조건을 복구한다.
2. 경로와 상태는 timestamp를 포함한 단일 타입 메시지로 전달한다.
3. 제어 입력 QoS는 최신값 우선 정책으로 검증한다.
4. 모든 시간 비교는 실행 모드별로 동일한 ROS clock 기준을 사용한다.
5. 정적 검사, rosbag 재현 시험, 바퀴를 띄운 상태의 dry-run 검증을 거친
   뒤에만 `enable_drive: true`로 전환한다.
