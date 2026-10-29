// [파일 역할] 번역 기록 화면이 쓰는 API: 목록(페이지 단위), 상세, 삭제. docs/api.md 번역 기록 절의 GET
// /api/history, GET·DELETE /api/history/{id}를 부른다.
// [연결] App.tsx가 api 클라이언트로 HistoryApi를 만들어 HistoryPage에 넘긴다.

import type { ApiClient } from '../api/client'
import { TranslationApi, type TranslationResult } from '../translation/api'

// 기록 한 건. 번역 응답과 같은 모양에서 tts_error만 빠진다. tts_error는 번역 응답에만 있고 기록에는 저장하지
// 않기 때문이다(docs/api.md).
export type HistoryItem = Omit<TranslationResult, 'tts_error'>
// 목록 응답. total은 페이지 크기와 관계없는 본인 기록 전체 개수라 화면이 다음 페이지가 있는지 계산하는 데
// 쓴다.
export interface HistoryList { items: HistoryItem[]; total: number }
// 한 페이지에 보이는 기록 수. 서버의 limit 기본값(20)과 같다. HistoryPage가 offset을 이 단위로 맞춘다.
export const PAGE_SIZE = 20
// 서버가 받는 offset의 최댓값(PostgreSQL integer 최댓값, docs/api.md GET /api/history). 넘으면 서버가 422로
// 거절하므로 HistoryPage가 주소의 offset을 이 값 안에서만 쓴다.
export const MAX_OFFSET = 2147483647

// TranslationApi를 이어받아 번역 음성 받기(audio) 같은 기능을 기록 화면에서도 그대로 쓴다. 부모의 client는
// private이라 자식에서 볼 수 없으므로 같은 객체를 historyClient로 한 번 더 들고 있는다.
export class HistoryApi extends TranslationApi {
  private readonly historyClient: Pick<ApiClient, 'request'>

  constructor(client: Pick<ApiClient, 'request'>) {
    super(client)
    this.historyClient = client
  }

  // 목록 한 페이지. 서버가 최신순(created_at, 같으면 id 내림차순)으로 정렬해 준다. signal로 페이지를 빨리
  // 넘길 때 앞 요청을 취소한다.
  async list(offset: number, signal: AbortSignal): Promise<HistoryList> {
    return (await this.historyClient.request(`/api/history?limit=${PAGE_SIZE}&offset=${offset}`, { signal })).json()
  }

  // 상세 한 건. 남의 기록과 없는 기록은 같은 404다. id는 주소 경로에서 오므로 encodeURIComponent로 감싸 다른
  // 경로로 바뀌지 않게 한다.
  async detail(id: string, signal: AbortSignal): Promise<HistoryItem> {
    return (await this.historyClient.request(`/api/history/${encodeURIComponent(id)}`, { signal })).json()
  }

  // 삭제(204). 서버가 기록과 연결된 번역 음성 파일까지 함께 지운다.
  async remove(id: string, signal: AbortSignal): Promise<void> {
    await this.historyClient.request(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE', signal })
  }
}
