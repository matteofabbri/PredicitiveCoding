# Vedere https://aka.ms/customizecontainer per informazioni su come personalizzare il contenitore di debug e su come Visual Studio usa questo Dockerfile per compilare le immagini per un debug più rapido.

# Questi argomenti consentono di scambiare la base usata per creare l'immagine finale durante il debug da Visual Studio
ARG LAUNCHING_FROM_VS
# In questo modo viene impostata l'immagine di base per la versione finale, ma solo se è stato definito LAUNCHING_FROM_VS
ARG FINAL_BASE_IMAGE=${LAUNCHING_FROM_VS:+aotdebug}

# Questa fase viene usata durante l'esecuzione da Visual Studio in modalità rapida (impostazione predefinita per la configurazione di debug)
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS base
USER $APP_UID
WORKDIR /app


# Questa fase viene usata per compilare il progetto di servizio
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
# Installare le dipendenze clang/zlib1g-dev per la pubblicazione in modalità nativa
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    clang zlib1g-dev
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["PredicitivCoding/PredicitivCoding.csproj", "PredicitivCoding/"]
RUN dotnet restore "./PredicitivCoding/PredicitivCoding.csproj"
COPY . .
WORKDIR "/src/PredicitivCoding"
RUN dotnet build "./PredicitivCoding.csproj" -c $BUILD_CONFIGURATION -o /app/build

# Questa fase viene usata per pubblicare il progetto di servizio da copiare nella fase finale
FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./PredicitivCoding.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=true

# Questa fase viene usata come base per la fase finale durante l'avvio da Visual Studio per supportare il debug in modalità normale (impostazione predefinita quando non si usa la configurazione di debug)
FROM base AS aotdebug
USER root
# Installare GDB per supportare il debug nativo
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gdb
USER app

# Questa fase viene usata nell'ambiente di produzione o durante l'esecuzione da Visual Studio in modalità normale (impostazione predefinita quando non si usa la configurazione di debug)
FROM ${FINAL_BASE_IMAGE:-mcr.microsoft.com/dotnet/runtime-deps:10.0} AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["./PredicitivCoding"]