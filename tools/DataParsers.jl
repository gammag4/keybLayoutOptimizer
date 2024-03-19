using JSON

# Maps telegram exported json data to a txt file
function mapTelegramJSONMessagesToTxt(jsonSrc, textDest)
    mapMessage(m) = (m isa Vector) ? join(map(i -> (i isa Dict) ? i["text"] : i, m), "\n") : m
    mapChat(chat) = map(i -> mapMessage(i["text"]), filter(i -> i["type"] == "message", collect(flatten(map(i -> map(j -> j, i["messages"]), chat["list"])))))
    mapChats(chats) = join(vcat(mapChat(chats["chats"]), mapChat(chats["left_chats"])), "\n")

    chats = mapChats(open(x -> JSON.parse(x), jsonSrc, "r"))
    open(x -> write(x, chats), textDest, "w")
end

mapTelegramJSONMessagesToTxt("result.json", "telegramData.txt")
