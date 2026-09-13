import type { ApiClient } from '../api/client'
import { TranslationApi, type TranslationResult } from '../translation/api'

export type HistoryItem = Omit<TranslationResult, 'tts_error'>
export interface HistoryList { items: HistoryItem[]; total: number }
export const PAGE_SIZE = 20

export class HistoryApi extends TranslationApi {
  private readonly historyClient: Pick<ApiClient, 'request'>

  constructor(client: Pick<ApiClient, 'request'>) {
    super(client)
    this.historyClient = client
  }

  async list(offset: number, signal: AbortSignal): Promise<HistoryList> {
    return (await this.historyClient.request(`/api/history?limit=${PAGE_SIZE}&offset=${offset}`, { signal })).json()
  }

  async detail(id: string, signal: AbortSignal): Promise<HistoryItem> {
    return (await this.historyClient.request(`/api/history/${encodeURIComponent(id)}`, { signal })).json()
  }

  async remove(id: string, signal: AbortSignal): Promise<void> {
    await this.historyClient.request(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE', signal })
  }
}
