export type DatabaseServerConfig = {
  url?: string | null
  hostname?: string | null
  port?: number
  database?: string | null
  user?: string | null
  password?: string | null
  type?: string | null
  schema?: string | null
  title?: string | null
}

export type ChatDatabaseServerConfig = {
  hostname?: string | null
  port?: number
  database?: string | null
  user?: string | null
  password?: string | null
  schema?: string | null
}

export type ConfigPayload = {
  database_server: DatabaseServerConfig
  chat_database_server: ChatDatabaseServerConfig
}

export type StoredConfig = {
  id: number
  db_url: string | null
  db_hostname: string | null
  db_port: number | null
  db_database: string | null
  db_user: string | null
  db_password: string | null
  db_type: string | null
  db_schema: string | null
  db_title: string | null
  chat_hostname: string | null
  chat_port: number | null
  chat_database: string | null
  chat_user: string | null
  chat_password: string | null
  chat_schema: string | null
}

export type ActiveConfigResponse = {
  active: (ConfigPayload & { id?: number }) | null
}

export type ExampleRow = {
  question_id: number
  type: string
  question_content: string
  generated: string | null
  public: boolean
  session_id: number | null
  query_id: number
  sql: string
  score: number | null
}

export type TableRow = {
  table: string
  description: string | null
}

export type ColumnRow = {
  table: string
  column: string
  description: string | null
  type: string
}

export type KnowledgeRow = {
  id: number
  reference: string
  content: string | null
}

export type KnowledgeUpsertPayload = {
  reference: string
  content: string
}

export type VectorstoreStatsResponse = {
  collections: Record<string, number>
}

export type VectorstoreSearchHit = {
  content: string
  metadata: Record<string, unknown>
  score: number
}

export type VectorstoreSearchResponse = {
  results: Record<string, VectorstoreSearchHit[]>
}

export type ChatHistoryRow = {
  id: number
  session_id: number
  sequence: number
  role: string
  timestamp: string
  content: string
}

export type ModelRow = {
  model_name: string
  provider: string | null
  reachable?: boolean
}

export type QueryResult = {
  columns: string[]
  rows: Array<Array<unknown>>
  count: number
}

export type StreamChatPayload = {
  prompt: string
  model_name?: string
  model_provider?: string | null
  session_id?: number | null
}

export type PlotCodePayload = {
  session_id: number
  model_name?: string
  model_provider?: string | null
  sql_override?: string
  plotting_request?: string
}

const API_BASE = (() => {
  const envBase = import.meta.env.VITE_API_BASE?.trim()
  if (envBase) {
    const normalized = envBase.startsWith('/') ? envBase : `/${envBase}`
    return normalized.replace(/\/$/, '')
  }

  const baseUrl = import.meta.env.BASE_URL?.trim() ?? '/'
  if (baseUrl === '/') {
    return ''
  }

  return baseUrl.replace(/\/$/, '')
})()

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }

  return (await response.json()) as T
}

async function requestStream(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }

  return response
}

export function toPayload(row: StoredConfig): ConfigPayload {
  return {
    database_server: {
      url: row.db_url,
      hostname: row.db_hostname,
      port: row.db_port ?? 5432,
      database: row.db_database,
      user: row.db_user,
      password: row.db_password,
      type: row.db_type ?? 'postgresql',
      schema: row.db_schema ?? 'public',
      title: row.db_title ?? 'Untitled',
    },
    chat_database_server: {
      hostname: row.chat_hostname,
      port: row.chat_port ?? 5432,
      database: row.chat_database,
      user: row.chat_user,
      password: row.chat_password,
      schema: row.chat_schema ?? 'public',
    },
  }
}

export const api = {
  listConfigs: () => request<{ items: StoredConfig[] }>('/api/settings/database-configs'),
  saveConfig: (payload: ConfigPayload) =>
    request<{ id: number }>('/api/settings/database-configs', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deleteConfig: (id: number) =>
    request<{ deleted: boolean }>(`/api/settings/database-configs/${id}`, {
      method: 'DELETE',
    }),
  getActive: () => request<ActiveConfigResponse>('/api/settings/database-configs/active'),
  selectConfig: (id: number) => request<{ active: ConfigPayload & { id?: number } }>(`/api/settings/database-configs/select/${id}`, { method: 'POST' }),
  connect: (payload: ConfigPayload & { save: boolean; create_chat_database_if_missing: boolean }) =>
    request<{ connected: boolean; active: ConfigPayload & { id?: number } }>('/api/settings/connect', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  listExamples: () => request<{ items: ExampleRow[] }>('/api/validator/examples'),
  validateExample: (questionId: number) =>
    request<{ validated: boolean; question_id: number }>(`/api/validator/examples/${questionId}/validate`, { method: 'POST' }),
  deleteExample: (questionId: number) =>
    request<{ deleted: boolean; question_id: number }>(`/api/validator/examples/${questionId}`, { method: 'DELETE' }),
  listTables: () => request<{ items: TableRow[] }>('/api/metadata/tables'),
  listColumns: () => request<{ items: ColumnRow[] }>('/api/metadata/columns'),
  listKnowledge: () => request<{ items: KnowledgeRow[] }>('/api/metadata/knowledge'),
  createKnowledge: (payload: KnowledgeUpsertPayload) =>
    request<{ saved: boolean; reference: string }>('/api/metadata/knowledge', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateKnowledge: (reference: string, payload: KnowledgeUpsertPayload) =>
    request<{ updated: boolean; reference: string }>(`/api/metadata/knowledge/${encodeURIComponent(reference)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteKnowledge: (knowledgeId: number) =>
    request<{ deleted: boolean; knowledge_id: number }>(`/api/metadata/knowledge/${knowledgeId}`, {
      method: 'DELETE',
    }),
  vectorstoreStats: () => request<VectorstoreStatsResponse>('/api/vectorstore/stats'),
  vectorstoreSearch: (payload: { query: string; collections?: string[]; k?: number }) =>
    request<VectorstoreSearchResponse>('/api/vectorstore/search', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  vectorstoreResync: () =>
    request<{ resynced: boolean; examples: number; knowledge: number }>('/api/vectorstore/resync', {
      method: 'POST',
    }),
  vectorstoreReset: () =>
    request<{ reset: boolean }>('/api/vectorstore/reset', {
      method: 'POST',
    }),
  listSessions: () => request<{ items: Array<{ session_id: number; timestamp: string; content: string }> }>('/api/chat/sessions'),
  createSession: (payload: {
    username: string
    email: string
    label?: string
    meta?: string
    referenced_session_id?: number | null
  }) =>
    request<{ session_id: number }>('/api/chat/sessions', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  getSessionHistory: (sessionId: number) => request<{ items: ChatHistoryRow[] }>(`/api/chat/sessions/${sessionId}/history`),
  undoSessionLastTurn: (sessionId: number) =>
    request<{ deleted: number; deleted_ids: number[] }>(`/api/chat/sessions/${sessionId}/undo`, {
      method: 'POST',
    }),
  saveMessagePair: (payload: {
    session_id: number
    user_prompt: string
    agent_response: string
    model_name?: string
  }) =>
    request<{ saved: boolean }>('/api/chat/messages', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  saveQuery: (payload: {
    session_id: number
    question_content: string
    sql: string
    question_type?: string
    public?: boolean
  }) =>
    request<{ saved: boolean }>('/api/chat/queries', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  saveQueryForValidation: (payload: {
    session_id: number
    sql: string
    model_name?: string
    model_provider?: string | null
    question_content?: string
    preview_only?: boolean
  }) =>
    request<{ saved: boolean; question_content: string }>('/api/chat/queries/prepare-save', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  runQuery: (payload: { sql: string }) =>
    request<QueryResult>('/api/chat/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  streamChat: (payload: StreamChatPayload, signal?: AbortSignal) =>
    requestStream('/api/chat/stream', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    }),
  generatePlotCode: (payload: PlotCodePayload) =>
    request<{ code: string; sql_used: string }>('/api/chat/plot-code', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  correctSql: (payload: { sql: string; error_message: string; history_summary?: string; model_name?: string; model_provider?: string | null }) =>
    request<{ corrected_sql: string; raw_response: string }>('/api/chat/correct-sql', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  listModels: () => request<{ items: ModelRow[] }>('/api/chat/models'),
  addModel: (payload: { model_name: string; provider?: string | null }) =>
    request<{ item: ModelRow }>('/api/chat/models', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deleteModel: (modelName: string, provider?: string | null) => {
    const params = new URLSearchParams({ model_name: modelName })
    if (provider) params.set('provider', provider)
    return request<{ deleted: boolean }>(`/api/chat/models?${params.toString()}`, { method: 'DELETE' })
  },
  runCode: (payload: { code: string; sql?: string }) =>
    request<{ plot_html: string | null; stdout: string; stderr: string; error: string | null }>(
      '/api/chat/run-code',
      { method: 'POST', body: JSON.stringify(payload) },
    ),
}
