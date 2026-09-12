import { useSyncExternalStore } from 'react'
import type { ApiClient } from '../api/client'

export function useSession(client: ApiClient) {
  return useSyncExternalStore(client.subscribe, client.getSnapshot)
}
