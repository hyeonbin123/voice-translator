/* global AudioWorkletProcessor, registerProcessor */
// [파일 역할] 마이크 소리를 오디오 처리 스레드에서 받아 메인 스레드로 넘기는 AudioWorklet 처리기. 대화 모드,
// 동시통역, 두 사람 대화가 모두 이것으로 소리를 받는다.
// [왜 AudioWorklet인가] 녹음 파일을 다 만든 뒤 넘기는 MediaRecorder와 달리, 소리를 렌더 단위(보통
// 128표본)마다 바로 받을 수 있어 말소리 판정과 실시간 전송에 쓸 수 있다. 오디오 전용 스레드에서 돌아 화면이
// 바빠도 소리가 끊기지 않는다.
// [연결] microphone.ts가 이 파일을 ?url로 가져와 audioWorklet.addModule로 올리고 'conversation-capture'
// 노드를 만든다. 맨 위 줄은 린터에게 이 두 이름이 워클릿 전역이라고 알려 주는 선언이다.

class ConversationCapture extends AudioWorkletProcessor {
  // enabled: 소리를 넘길지. epoch: 일시정지·재개를 바꿀 때마다 메인 스레드가 올리는 번호. 메인 스레드는
  // 바꾸기 전에 보낸 조각(번호가 다른 조각)을 버린다. 둘 다 메인 스레드가 port로 보내는 {paused, epoch}
  // 메시지로 바뀐다.
  constructor() {
    super()
    this.enabled = true
    this.epoch = 0
    this.port.onmessage = ({ data }) => {
      this.enabled = !data.paused
      this.epoch = data.epoch
    }
  }
  // 렌더 단위마다 불린다. 첫 입력의 채널들을 평균해 모노로 섞고, 버퍼 소유권을 넘기는(transfer) 방식으로 보내
  // 복사 없이 메인 스레드로 옮긴다. 일시정지 중이거나 입력이 없으면 보내지 않는다. 표본화율은 브라우저
  // AudioContext의 값(보통 44.1·48kHz) 그대로이고, 16kHz로 바꾸는 일은 메인 스레드의 MonoResampler(pcm.ts)가
  // 한다.
  process(inputs) {
    const channels = inputs[0]
    if (this.enabled && channels?.length) {
      const samples = new Float32Array(channels[0].length)
      for (const channel of channels) {
        for (let i = 0; i < samples.length; i++) samples[i] += channel[i] / channels.length
      }
      this.port.postMessage({ samples, epoch: this.epoch }, [samples.buffer])
    }
    // true를 돌려줘야 입력이 잠시 비어도 처리기가 살아 있다.
    // Output stays silent: microphone audio never goes to the speakers.
    return true
  }
}
registerProcessor('conversation-capture', ConversationCapture)
