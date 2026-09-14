import type { ApiClient } from '../api/client'

export type Language = 'ko' | 'en'
export interface Direction { source_lang: Language; target_lang: Language }
export interface TranslationResult extends Direction {
  id: string
  mode: 'text' | 'speech'
  source_text: string
  translated_text: string
  stt_model: string | null
  mt_model: string
  tts_model: string | null
  stt_ms: number | null
  mt_ms: number
  tts_ms: number | null
  audio_id: string | null
  created_at: string
  tts_error: 'Speech synthesis failed' | 'Speech synthesis is not available' | null
}
export interface DialogTranslationResult extends TranslationResult {
  language_confidence: number
  language_guessed: boolean
}

export const MAX_AUDIO_BYTES = 10_000_000 // Conservative decimal MB; server validates decoded duration.
export const characterCount = (text: string) => Array.from(text.trim()).length

export class TranslationApi {
  private readonly client: Pick<ApiClient, 'request'> & Partial<Pick<ApiClient, 'liveToken' | 'logout'>>
  readonly demo: boolean

  constructor(client: Pick<ApiClient, 'request'> & Partial<Pick<ApiClient, 'liveToken' | 'logout'>>, demo = false) {
    this.client = client
    this.demo = demo
  }

  async liveToken(rejectedToken?: string): Promise<string> {
    if (this.demo || !this.client.liveToken) throw new Error('Live translation needs authentication')
    return this.client.liveToken(rejectedToken)
  }

  expireLiveAuthentication() { this.client.logout?.(true) }

  async text(text: string, direction: Direction, signal: AbortSignal): Promise<TranslationResult> {
    return (await this.client.request('/api/translate/text', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.trim(), ...direction }), signal,
    })).json()
  }

  async speech(audio: Blob, direction: Direction, signal: AbortSignal): Promise<TranslationResult> {
    const body = new FormData()
    const extension = audio.type.includes('ogg') ? 'ogg' : audio.type.includes('mp4') ? 'm4a' : 'webm'
    body.append('audio', audio, audio instanceof File ? audio.name : `recording.${extension}`)
    body.append('source_lang', direction.source_lang)
    body.append('target_lang', direction.target_lang)
    return (await this.client.request('/api/translate/speech', { method: 'POST', body, signal })).json()
  }

  async dialog(audio: Blob, previousLang: Language | undefined, signal: AbortSignal): Promise<DialogTranslationResult> {
    const body = new FormData()
    body.append('audio', audio, audio instanceof File ? audio.name : 'conversation.wav')
    if (previousLang) body.append('previous_lang', previousLang)
    return (await this.client.request('/api/translate/dialog', { method: 'POST', body, signal })).json()
  }

  async removeHistory(id: string, signal: AbortSignal): Promise<void> {
    await this.client.request(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE', signal })
  }

  async audio(id: string, signal: AbortSignal): Promise<Blob> {
    return (await this.client.request(`/api/audio/${encodeURIComponent(id)}`, { signal })).blob()
  }
}
