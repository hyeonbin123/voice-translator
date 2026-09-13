/* global AudioWorkletProcessor, registerProcessor */
class ConversationCapture extends AudioWorkletProcessor {
  constructor() {
    super()
    this.enabled = true
    this.epoch = 0
    this.port.onmessage = ({ data }) => {
      this.enabled = !data.paused
      this.epoch = data.epoch
    }
  }
  process(inputs) {
    const channels = inputs[0]
    if (this.enabled && channels?.length) {
      const samples = new Float32Array(channels[0].length)
      for (const channel of channels) {
        for (let i = 0; i < samples.length; i++) samples[i] += channel[i] / channels.length
      }
      this.port.postMessage({ samples, epoch: this.epoch }, [samples.buffer])
    }
    // Output stays silent: microphone audio never goes to the speakers.
    return true
  }
}
registerProcessor('conversation-capture', ConversationCapture)
