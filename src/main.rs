#![allow(unused_labels)]
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use std::{env, iter, sync::Arc};

use anyhow::Result;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use async_openai::{Client as OAIClient, config::OpenAIConfig};
use const_format::formatcp;
use grammers_client::grammers_tl_types::enums::SendMessageAction;
use grammers_client::types::Chat;
use grammers_client::{
    Client, Config, FixedReconnect, InitParams, InputMessage, SignInError, Update, session::Session,
};
use inquire::Text;
use rand::Rng;
use tokio::{sync::Mutex, time::sleep};

const APP_VERSION: &str = "0.1.0";

const WHITELIST_MODE: bool = true;
const WHITELISTED_CHATS: [i64; 2] = [1262613096, 1154500568];
const WHITELISTED_COMMENT_CHATS: [i64; 3] = [1262613096, 3049292659, 1154500568];
const WHITELIST_REACTION_RATE: f64 = 0.1;
const ALWAYS_REACT_CHATS: [i64; 3] = [3052201490, 1529721824, 1624587827];

const RECONNECTION_POLICY: FixedReconnect = FixedReconnect {
    attempts: 100,
    delay: Duration::from_secs(5),
};

const MAX_RETRY: usize = 2;
const MAX_TOKENS: u32 = 2000;
const CONTEXT_MESSAGES_PER_CHAT: usize = 50;
const MINIMUM_CONTEXT_MESSAGES: usize = 20;
const OPENAI_MODEL: &str = "deepseek-v3.1";
const SESSION_FILE: &str = ".ruiping-session";

const DOCUMENTATION: &str = formatcp!(
    "Ruiping version {APP_VERSION}
全自动水群、频道评论 https://github.com/arielherself/ruiping

支持指令：
  `rep` 一键复读
"
);

macro_rules! retry_future {
    ($future:expr) => {{
        let mut result = $future.await;
        if matches!(result, Err(_)) {
            for i in 1..MAX_RETRY {
                log::debug!("Retrying: {}/{MAX_RETRY}", i + 1);
                sleep(Duration::from_secs(2)).await;
                let new_result = $future.await;
                if matches!(new_result, Ok(_)) {
                    result = new_result;
                    break;
                }
            }
        }
        result
    }};
}

#[inline]
fn get_env(name: &str) -> Result<String> {
    env::var(name).map_err(|_| anyhow::anyhow!("please provide {name}"))
}

#[derive(Clone, Debug)]
struct Message {
    username: String,
    content: String,
}

impl Message {
    async fn from(value: &grammers_client::types::Message) -> Result<Option<Self>> {
        let (name, username) = match value.sender().unwrap() {
            Chat::User(user) => (
                user.full_name(),
                user.username()
                    .unwrap_or(&user.id().to_string())
                    .to_string(),
            ),
            Chat::Channel(channel) => (
                channel.title().to_string(),
                channel
                    .username()
                    .unwrap_or(&channel.id().to_string())
                    .to_string(),
            ),
            _ => return Ok(None),
        };
        let mut content_prefix = format!("[Name: {name}]");
        if let Some(origin) = value.get_reply().await? {
            let origin_sender = match origin.sender().unwrap() {
                Chat::User(user) => user,
                _ => return Ok(None),
            };
            content_prefix += &format!("[Replied to \"{}\"]", origin_sender.full_name());
        };
        if value.media().is_some() {
            content_prefix += "[Sent a media]";
        }
        Ok(Some(Self {
            username,
            content: format!("{content_prefix}{}", value.text()),
        }))
    }
}

struct ChatClient {
    config: OpenAIConfig,
    name: String,
    username: String,
    http_client: reqwest::Client,
}

impl ChatClient {
    fn new(config: OpenAIConfig, name: &str, username: &str) -> Self {
        Self {
            config,
            name: name.to_string(),
            username: username.to_string(),
            http_client: reqwest::Client::builder()
                .http1_only()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        }
    }
    async fn query(&self, ctx: &[Message]) -> Result<String> {
        let oai_client =
            OAIClient::with_config(self.config.clone()).with_http_client(self.http_client.clone());
        let oai_messages = iter::once(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(format!(
                    "You should reply on behalf of the user \"{}\" (username @\"{}\"), and give a precise reply to the last message in 1~2 sentences in the language of the chat. Your tone should be very casual, and you are a human being. You are a common group member, and does not play any role. Your response should be interesting and closely related to the topic, but should have quality and DO NOT REPEAT the content of the previous chat.",
                    self.name, self.username
                ))
                .build()
                .unwrap(),
        ))
        .chain(ctx.iter().map(|msg| {
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .name(&msg.username)
                    .content(&*msg.content)
                    .build()
                    .unwrap(),
            )
        }))
        .collect::<Vec<_>>();

        log::debug!("Request: {:#?}", oai_messages);

        let oai_request = CreateChatCompletionRequestArgs::default()
            .model(OPENAI_MODEL)
            .messages(oai_messages)
            .max_tokens(MAX_TOKENS)
            .build()?;

        for _ in 0..MAX_RETRY {
            let response = oai_client.chat().create(oai_request.clone()).await?;

            let content = response
                .choices
                .first()
                .unwrap()
                .message
                .content
                .clone()
                .unwrap()
                .trim()
                .to_string();

            if !content.is_empty() {
                return Ok(content);
            }

            log::warn!("Empty message is generated");
        }

        Ok(String::new())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let openai_api_key = get_env("OPENAI_API_KEY")?;
    let openai_api_base = get_env("OPENAI_API_BASE")?;
    let oai_config = OpenAIConfig::new()
        .with_api_key(openai_api_key)
        .with_api_base(openai_api_base);

    let api_id = get_env("TELEGRAM_API_ID")?.parse().expect("invalid api id");
    let api_hash = get_env("TELEGRAM_API_HASH")?;

    let client = Arc::new(
        Client::connect(Config {
            session: Session::load_file_or_create(SESSION_FILE)?,
            api_id,
            api_hash,
            params: InitParams {
                device_model: String::from("Ruiping Headless"),
                system_version: String::from(APP_VERSION),
                app_version: String::from(APP_VERSION),
                system_lang_code: String::from("C"),
                lang_code: String::from("C"),
                catch_up: false,
                server_addr: None,
                flood_sleep_threshold: 60,
                update_queue_limit: Some(100),
                reconnection_policy: &RECONNECTION_POLICY,
            },
        })
        .await?,
    );
    log::info!("Connected to server");

    if !client.is_authorized().await? {
        log::info!("Signing in");
        let phone_number = Text::new("Enter your phone number (internation format):").prompt()?;
        let login_token = client.request_login_code(&phone_number).await?;
        let login_code = Text::new("Enter the code you received:").prompt()?;
        let signed_in = client.sign_in(&login_token, &login_code).await;
        match signed_in {
            Err(SignInError::PasswordRequired(password_token)) => {
                let hint = password_token.hint().unwrap();
                let prompt_message = format!("Enter the password (hint: {hint}):");
                let password = Text::new(&prompt_message).prompt()?;
                client.check_password(password_token, password.trim()).await
            }
            x @ _ => x,
        }?;
        client.session().save_to_file(SESSION_FILE)?;
    }

    let me = client.get_me().await?;
    let my_username = me.username().unwrap();
    let my_name = me.full_name();
    log::info!("Signed in as @{} ({})", my_username, my_name);
    let chat_client = Arc::new(ChatClient::new(oai_config, &my_name, &my_username));

    let chat_storage = Arc::new(Mutex::new(HashMap::<i64, VecDeque<Message>>::new()));

    let handle_update = |update: Update| {
        let me = me.clone();
        let client = Arc::clone(&client);
        let chat_client = Arc::clone(&chat_client);
        let chat_storage = Arc::clone(&chat_storage);
        async move {
            match update {
                Update::NewMessage(msg) => {
                    if matches!(msg.chat(), Chat::Channel(_)) {
                        return Ok(());
                    }

                    log::debug!("New message {}: {}", msg.id(), msg.text());

                    let message = Message::from(&msg).await?;
                    let mut message_number = 0;
                    if let Some(message) = message {
                        let mut guard = chat_storage.lock().await;
                        let v = guard
                            .entry(msg.chat().id())
                            .or_insert(VecDeque::with_capacity(CONTEXT_MESSAGES_PER_CHAT));
                        v.push_back(message);
                        if v.len() > CONTEXT_MESSAGES_PER_CHAT {
                            v.pop_front();
                        }
                        message_number = v.len();
                        drop(guard);
                    }

                    'command_handler: {
                        let text = msg.text();
                        if !msg.outgoing() || !text.starts_with(',') {
                            break 'command_handler;
                        }
                        let separator = text.find(' ').unwrap_or(text.len());
                        let cmd = &text[1..separator];
                        let arg = if separator == text.len() {
                            ""
                        } else {
                            &text[separator + 1..]
                        };
                        let response_text = match cmd {
                            "info" => String::from(DOCUMENTATION),
                            "rep" => {
                                let origin = msg.get_reply().await?;
                                match origin {
                                    Some(origin) => {
                                        msg.delete().await?;
                                        client
                                            .forward_messages(
                                                msg.chat(),
                                                &[origin.id()],
                                                origin.chat(),
                                            )
                                            .await?;
                                        String::new()
                                    }
                                    None => String::from("请回复要复读的消息"),
                                }
                            }
                            _ => format!("未知指令 `{cmd}`"),
                        };
                        if !response_text.is_empty() {
                            msg.edit(InputMessage::text(response_text)).await?;
                        }
                    }

                    'ai_response: {
                        let group_id = match msg.chat() {
                            Chat::Group(group) => Some(group.id()),
                            _ => None,
                        };

                        let r = rand::rng().random::<u8>();
                        let p = r as f64 / u8::max_value() as f64;

                        let is_always_react =
                            group_id.is_some_and(|group_id| ALWAYS_REACT_CHATS.contains(&group_id));
                        let is_myself = msg.sender().unwrap().id() == me.id();

                        if msg.mentioned()
                            || matches!(msg.chat(), Chat::User(_))
                            || is_always_react && !is_myself
                            || (WHITELIST_MODE == false
                                || group_id
                                    .is_some_and(|group_id| WHITELISTED_CHATS.contains(&group_id)))
                                && !is_myself
                                && p < WHITELIST_REACTION_RATE
                                && message_number > MINIMUM_CONTEXT_MESSAGES
                            || group_id.is_some_and(|group_id| {
                                WHITELISTED_COMMENT_CHATS.contains(&group_id)
                            }) && matches!(msg.sender().unwrap(), Chat::Channel(_))
                        {
                            if is_always_react {
                                msg.react("👀").await?;
                            }

                            let v = {
                                let mut guard = chat_storage.lock().await;
                                let r = guard
                                    .entry(msg.chat().id())
                                    .or_insert(VecDeque::with_capacity(CONTEXT_MESSAGES_PER_CHAT));
                                let v = r.clone();
                                r.clear();
                                drop(guard);
                                v.into_iter().collect::<Vec<_>>()
                            };

                            let generate_response = async {
                                let mut response = chat_client.query(&v).await?;
                                if response.is_empty() {
                                    response = String::from("。。。");
                                }

                                retry_future!(msg.reply(InputMessage::text(&response)))?;

                                log::info!("Sent response: {response}");

                                Result::<()>::Ok(())
                            };
                            tokio::pin!(generate_response);

                            client
                                .action(msg.chat())
                                .repeat(
                                    || SendMessageAction::SendMessageTypingAction,
                                    generate_response,
                                )
                                .await
                                .0?;
                        }
                    }
                }
                Update::MessageEdited(msg) => {
                    log::debug!("Message edited {}: {}", msg.id(), msg.text());
                }
                Update::MessageDeleted(msg) => {
                    log::debug!("Message deleted {:?}", msg.into_messages());
                }
                _ => (),
            }
            Result::<()>::Ok(())
        }
    };

    loop {
        let update = client.next_update().await;
        if !update.is_ok() {
            continue;
        }
        let update = update.unwrap();
        tokio::spawn(handle_update(update));
    }
}
